#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <chrono>
#include <string>
#include <sstream>
#include <algorithm>
#include <numeric>   // For std::iota
#include <random>    // For std::shuffle
#include <numeric> // For std::accumulate

// 编译文件
// hipcc lesson3_sourcefile_mlp.cpp -o mlp_full_dcu
// 执行文件
// ./mlp_full_dcu 或者 hipprof ./mlp_full_dcu

// 预定义参数，可根据需求修改
#define INPUT_DIM 10
#define HIDDEN_DIM1 256
#define HIDDEN_DIM2 128  // 第二个隐藏层
#define HIDDEN_DIM3 64   // 第三个隐藏层（可选）
#define OUTPUT_DIM 1
#define BATCH_SIZE 256
#define EPOCHS 700
#define LEARNING_RATE 1e-3

#define HIP_CHECK(error) \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: %s (%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// ----------------------------- HIP Kernels -------------------------------

// Matrix multiplication: C = A * B
// A: M x K, B: K x N, C: M x N
__global__ void matmul_kernel(const double* A, const double* B, double* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        double sum = 0.0;
        for (int i = 0; i < K; ++i) {
            sum += A[row * K + i] * B[i * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Transpose matrix: Output = Input^T
// Input: R x C, Output: C x R
__global__ void transpose_kernel(const double* input, double* output, int R, int C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // row in input, col in output
    int col = blockIdx.x * blockDim.x + threadIdx.x; // col in input, row in output

    if (row < R && col < C) {
        output[col * R + row] = input[row * C + col];
    }
}

// Add bias to a matrix (broadcasts bias vector)
// matrix: M_rows x N_cols, bias: N_cols
__global__ void add_bias_kernel(double* matrix, const double* bias, int M_rows, int N_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_rows && col < N_cols) {
        matrix[row * N_cols + col] += bias[col];
    }
}

// ReLU forward activation
__global__ void relu_forward_kernel(double* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmax(0.0, data[idx]);
    }
}

// Gradient of MSE loss w.r.t. prediction: grad = pred - target
__global__ void compute_output_grad(const double* pred, const double* target, double* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = pred[idx] - target[idx];
    }
}

// ReLU backward: grad_input = (pre_activation > 0) ? grad_output : 0
// delta is grad_output (input) and becomes grad_input (output)
// activ is pre-activation values
__global__ void compute_relu_backward(double* delta, const double* activ, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        delta[idx] = (activ[idx] > 0.0) ? delta[idx] : 0.0;
    }
}

// Compute element-wise squared error for MSE loss: loss_elements[i] = (pred[i] - target[i])^2
__global__ void compute_mse_loss_elements(const double* pred, const double* target, double* loss_elements, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        double diff = pred[idx] - target[idx];
        loss_elements[idx] = diff * diff;
    }
}

// SGD weight update: weights = weights - lr * grad
__global__ void sgd_update(double* weights, const double* grad, double lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

// ----------------------------- CPU Helper Functions -------------------------------

// Load bandwidth data from comma-separated file
std::vector<double> load_bandwidth_data(const std::string& filename) {
    std::vector<double> data;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string line;
    std::getline(file, line); // Assuming all data is on a single line
    file.close();

    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) {
        try {
            data.push_back(std::stod(item));
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << ia.what() << " for item: " << item << std::endl;
        } catch (const std::out_of_range& oor) {
            std::cerr << "Out of Range error: " << oor.what() << " for item: " << item << std::endl;
        }
    }
    std::cout << "[INFO] Loaded " << data.size() << " bandwidth data points." << std::endl;
    return data;
}

// Create dataset using sliding window
// X will store sequences of INPUT_DIM, y will store the next value
void create_dataset(const std::vector<double>& data,
                    std::vector<double>& X,
                    std::vector<double>& y,
                    int window_size) {
    X.clear();
    y.clear();
    if (data.size() <= window_size) {
        std::cerr << "Error: Data size is too small for the window size." << std::endl;
        return;
    }
    for (size_t i = 0; i < data.size() - window_size; ++i) {
        for (int j = 0; j < window_size; ++j) {
            X.push_back(data[i + j]);
        }
        y.push_back(data[i + window_size]);
    }
    std::cout << "[INFO] Created dataset with " << y.size() << " samples." << std::endl;
}

// Data normalization (min-max scaling to [0, 1])
void normalize_data(std::vector<double>& data, double& min_val, double& max_val) {
    if (data.empty()) return;
    min_val = *std::min_element(data.begin(), data.end());
    max_val = *std::max_element(data.begin(), data.end());
    if (max_val == min_val) { // Avoid division by zero if all elements are the same
        for (auto& val : data) {
            val = 0.5; // Or 0, or 1, depending on preference
        }
        return;
    }
    for (auto& val : data) {
        val = (val - min_val) / (max_val - min_val);
    }
}

// Data denormalization
void denormalize_data(std::vector<double>& data, double min_val, double max_val) {
    if (max_val == min_val) { // Consistent with normalization
         for (auto& val : data) {
            val = min_val; // Or map 0.5 back to the constant value
        }
        return;
    }
    for (auto& val : data) {
        val = val * (max_val - min_val) + min_val;
    }
}

// Initialize weights with small random values
void random_initialize_weights(std::vector<double>& vec, unsigned int seed) {
    std::mt19937 gen(seed);
    // He initialization for ReLU: stddev = sqrt(2.0 / fan_in)
    // For simplicity, using a small uniform range.
    // A proper fan_in based initialization would be better.
    std::uniform_real_distribution<> distrib(-0.1, 0.1);
    for (auto& val : vec) {
        val = distrib(gen);
    }
}

// Sum vector elements (for bias gradients)
double sum_vector_elements(const std::vector<double>& vec) {
    return std::accumulate(vec.begin(), vec.end(), 0.0);
}

// ----------------------------- Main -------------------------------
int main() {
    auto global_start_time = std::chrono::high_resolution_clock::now();

    // --- 1. Data Preparation ---
    std::vector<double> raw_bandwidth_data = load_bandwidth_data("source.txt"); // Ensure source.txt is named starlink_bw.json or change here
    if (raw_bandwidth_data.empty()) {
        return 1;
    }

    double min_bw, max_bw;
    std::vector<double> normalized_data = raw_bandwidth_data;
    normalize_data(normalized_data, min_bw, max_bw);

    std::vector<double> X_all, y_all;
    create_dataset(normalized_data, X_all, y_all, INPUT_DIM);

    if (X_all.empty()) {
        std::cerr << "Dataset creation failed or resulted in empty dataset." << std::endl;
        return 1;
    }
    
    size_t num_samples = y_all.size();
    size_t train_size = static_cast<size_t>(num_samples * 0.8);
    size_t test_size = num_samples - train_size;

    if (train_size == 0 || test_size == 0) {
        std::cerr << "Not enough samples for train/test split. Need at least " << 5 * (INPUT_DIM +1) << " data points for 80/20 split." << std::endl;
        return 1;
    }


    std::vector<double> X_train(X_all.begin(), X_all.begin() + train_size * INPUT_DIM);
    std::vector<double> y_train(y_all.begin(), y_all.begin() + train_size);
    std::vector<double> X_test(X_all.begin() + train_size * INPUT_DIM, X_all.end());
    std::vector<double> y_test(y_all.begin() + train_size, y_all.end());

    std::cout << "[INFO] Training samples: " << train_size << ", Test samples: " << test_size << std::endl;
// --- 2. MLP Initialization (Host) ---
unsigned int random_seed = 42;
srand(random_seed);

// 原有的第一层权重和偏置
std::vector<double> h_W1(INPUT_DIM * HIDDEN_DIM1);
std::vector<double> h_B1(HIDDEN_DIM1);

// 新增的第二层权重和偏置
std::vector<double> h_W2(HIDDEN_DIM1 * HIDDEN_DIM2);
std::vector<double> h_B2(HIDDEN_DIM2);

// 新增的第三层权重和偏置（可选）
std::vector<double> h_W3(HIDDEN_DIM2 * HIDDEN_DIM3);
std::vector<double> h_B3(HIDDEN_DIM3);

// 原来的输出层权重和偏置（现在是W4和B4）
std::vector<double> h_W4(HIDDEN_DIM3 * OUTPUT_DIM);
std::vector<double> h_B4(OUTPUT_DIM);

// 初始化权重
random_initialize_weights(h_W1, random_seed);
std::fill(h_B1.begin(), h_B1.end(), 0.0);

random_initialize_weights(h_W2, random_seed + 1);
std::fill(h_B2.begin(), h_B2.end(), 0.0);

random_initialize_weights(h_W3, random_seed + 2);
std::fill(h_B3.begin(), h_B3.end(), 0.0);

random_initialize_weights(h_W4, random_seed + 3);
std::fill(h_B4.begin(), h_B4.end(), 0.0);

    // --- 3. Device Memory Allocation ---
    double *d_W1, *d_B1, *d_W2, *d_B2;
    double *d_X_batch, *d_Y_batch, *d_Y_pred;
    double *d_Z1, *d_A1, *d_Z2; // Intermediate activations/pre-activations
    double *d_grad_W1, *d_grad_B1, *d_grad_W2, *d_grad_B2; // Gradients for weights/biases
    double *d_dZ2, *d_dA1, *d_dZ1; // Gradients for activations/pre-activations
    double *d_A1_T, *d_X_batch_T, *d_W2_T; // For transposed matrices
    double *d_mse_elements;


    HIP_CHECK(hipMalloc(&d_W1, INPUT_DIM * HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B1, HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_B2, OUTPUT_DIM * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_X_batch, BATCH_SIZE * INPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Y_batch, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Y_pred, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));
    
    HIP_CHECK(hipMalloc(&d_Z1, BATCH_SIZE * HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_A1, BATCH_SIZE * HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_Z2, BATCH_SIZE * OUTPUT_DIM * sizeof(double))); // Same as d_Y_pred if linear output

    HIP_CHECK(hipMalloc(&d_grad_W1, INPUT_DIM * HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_B1, HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_W2, HIDDEN_DIM * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_grad_B2, OUTPUT_DIM * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_dZ2, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_dA1, BATCH_SIZE * HIDDEN_DIM * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_dZ1, BATCH_SIZE * HIDDEN_DIM * sizeof(double)));

    HIP_CHECK(hipMalloc(&d_A1_T, HIDDEN_DIM * BATCH_SIZE * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_X_batch_T, INPUT_DIM * BATCH_SIZE * sizeof(double)));
    HIP_CHECK(hipMalloc(&d_W2_T, OUTPUT_DIM * HIDDEN_DIM * sizeof(double)));
    
    HIP_CHECK(hipMalloc(&d_mse_elements, BATCH_SIZE * OUTPUT_DIM * sizeof(double)));


    // Copy initial weights to device
    HIP_CHECK(hipMemcpy(d_W1, h_W1.data(), INPUT_DIM * HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B1, h_B1.data(), HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_W2, h_W2.data(), HIDDEN_DIM * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B2, h_B2.data(), OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));

    // Kernel launch parameters
    dim3 threads_per_block_2d(16, 16);
    dim3 threads_per_block_1d(256);


    // --- 4. Training Loop ---
    std::cout << "\n[INFO] Starting Training..." << std::endl;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        auto epoch_start_time = std::chrono::high_resolution_clock::now();
        double epoch_loss = 0.0;
        int num_batches = (train_size + BATCH_SIZE - 1) / BATCH_SIZE;

        // Simple shuffle of training indices per epoch
        std::vector<size_t> train_indices(train_size);
        std::iota(train_indices.begin(), train_indices.end(), 0);
        std::shuffle(train_indices.begin(), train_indices.end(), std::mt19937(std::random_device()()));


        for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
            size_t current_batch_offset = batch_idx * BATCH_SIZE;
            size_t current_batch_size = std::min((size_t)BATCH_SIZE, train_size - current_batch_offset);
            if (current_batch_size == 0) continue;

            // Prepare batch data on host
            std::vector<double> h_X_batch_vec(current_batch_size * INPUT_DIM);
            std::vector<double> h_Y_batch_vec(current_batch_size * OUTPUT_DIM);

            for(size_t i=0; i<current_batch_size; ++i) {
                size_t sample_idx = train_indices[current_batch_offset + i];
                for(int j=0; j<INPUT_DIM; ++j) {
                    h_X_batch_vec[i * INPUT_DIM + j] = X_train[sample_idx * INPUT_DIM + j];
                }
                for(int j=0; j<OUTPUT_DIM; ++j) { // OUTPUT_DIM is 1
                     h_Y_batch_vec[i * OUTPUT_DIM + j] = y_train[sample_idx * OUTPUT_DIM + j];
                }
            }
            
            HIP_CHECK(hipMemcpy(d_X_batch, h_X_batch_vec.data(), current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_Y_batch, h_Y_batch_vec.data(), current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));

            // --- Forward Pass ---
            // Layer 1: Z1 = X_batch * W1
            dim3 blocks_matmul_z1((HIDDEN_DIM + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                                  (current_batch_size + threads_per_block_2d.y - 1) / threads_per_block_2d.y);
            matmul_kernel<<<blocks_matmul_z1, threads_per_block_2d>>>(d_X_batch, d_W1, d_Z1, current_batch_size, HIDDEN_DIM, INPUT_DIM);
            
            // Add bias B1: Z1_biased = Z1 + B1
            // d_Z1 is used for Z1_biased to save memory, add_bias_kernel adds in place
            HIP_CHECK(hipMemcpy(d_A1, d_Z1, current_batch_size * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToDevice)); // Copy Z1 to A1 (which will become Z1_biased)
            add_bias_kernel<<<blocks_matmul_z1, threads_per_block_2d>>>(d_A1, d_B1, current_batch_size, HIDDEN_DIM); // d_A1 now holds Z1_biased
            
            // Activation A1 = ReLU(Z1_biased)
            // d_A1 is used for Z1_biased (input) and A1 (output)
            dim3 blocks_relu_a1((current_batch_size * HIDDEN_DIM + threads_per_block_1d.x - 1) / threads_per_block_1d.x);
            relu_forward_kernel<<<blocks_relu_a1, threads_per_block_1d>>>(d_A1, current_batch_size * HIDDEN_DIM); // d_A1 now holds activated A1

            // Layer 2: Z2 = A1 * W2
            dim3 blocks_matmul_z2((OUTPUT_DIM + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                                  (current_batch_size + threads_per_block_2d.y - 1) / threads_per_block_2d.y);
            matmul_kernel<<<blocks_matmul_z2, threads_per_block_2d>>>(d_A1, d_W2, d_Z2, current_batch_size, OUTPUT_DIM, HIDDEN_DIM);

            // Add bias B2: Y_pred = Z2 + B2 (linear output layer)
            // d_Y_pred will store the final prediction
            HIP_CHECK(hipMemcpy(d_Y_pred, d_Z2, current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToDevice));
            add_bias_kernel<<<blocks_matmul_z2, threads_per_block_2d>>>(d_Y_pred, d_B2, current_batch_size, OUTPUT_DIM);

            // --- Backward Pass ---
            // Gradient of loss w.r.t. Y_pred (dZ2 if linear output): dL/dY_pred = Y_pred - Y_batch
            dim3 blocks_out_grad((current_batch_size * OUTPUT_DIM + threads_per_block_1d.x - 1) / threads_per_block_1d.x);
            compute_output_grad<<<blocks_out_grad, threads_per_block_1d>>>(d_Y_pred, d_Y_batch, d_dZ2, current_batch_size * OUTPUT_DIM);

            // Gradient for W2: dW2 = A1^T * dZ2
            dim3 blocks_transpose_A1((current_batch_size + threads_per_block_2d.x -1) / threads_per_block_2d.x,
                                     (HIDDEN_DIM + threads_per_block_2d.y -1) / threads_per_block_2d.y);
            transpose_kernel<<<blocks_transpose_A1, threads_per_block_2d>>>(d_A1, d_A1_T, current_batch_size, HIDDEN_DIM); // A1 is (batch, hidden) -> A1_T is (hidden, batch)
            
            dim3 blocks_matmul_dw2((OUTPUT_DIM + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                                   (HIDDEN_DIM + threads_per_block_2d.y - 1) / threads_per_block_2d.y);
            matmul_kernel<<<blocks_matmul_dw2, threads_per_block_2d>>>(d_A1_T, d_dZ2, d_grad_W2, HIDDEN_DIM, OUTPUT_DIM, current_batch_size);

            // Gradient for B2: sum dZ2 over batch dimension
            // For simplicity, copy d_dZ2 to host and sum. A reduction kernel would be better.
            std::vector<double> h_dZ2_vec(current_batch_size * OUTPUT_DIM);
            HIP_CHECK(hipMemcpy(h_dZ2_vec.data(), d_dZ2, current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
            std::vector<double> h_grad_B2_vec(OUTPUT_DIM, 0.0);
            for(size_t i=0; i < current_batch_size; ++i) {
                for(int j=0; j < OUTPUT_DIM; ++j) {
                    h_grad_B2_vec[j] += h_dZ2_vec[i * OUTPUT_DIM + j];
                }
            }
            HIP_CHECK(hipMemcpy(d_grad_B2, h_grad_B2_vec.data(), OUTPUT_DIM * sizeof(double), hipMemcpyHostToDevice));


            // Gradient dA1 = dZ2 * W2^T
            dim3 blocks_transpose_W2((HIDDEN_DIM + threads_per_block_2d.x -1) / threads_per_block_2d.x,
                                     (OUTPUT_DIM + threads_per_block_2d.y -1) / threads_per_block_2d.y);
            transpose_kernel<<<blocks_transpose_W2, threads_per_block_2d>>>(d_W2, d_W2_T, HIDDEN_DIM, OUTPUT_DIM); // W2 is (hidden, out) -> W2_T is (out, hidden)
            
            dim3 blocks_matmul_da1((HIDDEN_DIM + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                                   (current_batch_size + threads_per_block_2d.y - 1) / threads_per_block_2d.y);
            matmul_kernel<<<blocks_matmul_da1, threads_per_block_2d>>>(d_dZ2, d_W2_T, d_dA1, current_batch_size, HIDDEN_DIM, OUTPUT_DIM);

           
            HIP_CHECK(hipMemcpy(d_dZ1, d_Z1, current_batch_size*HIDDEN_DIM*sizeof(double), hipMemcpyDeviceToDevice)); // d_Z1 is X*W1
            add_bias_kernel<<<blocks_matmul_z1, threads_per_block_2d>>>(d_dZ1, d_B1, current_batch_size, HIDDEN_DIM); // d_dZ1 now Z1_biased
            // d_dA1 is the incoming gradient. d_dZ1 is Z1_biased. compute_relu_backward modifies d_dA1 in place.
            compute_relu_backward<<<blocks_relu_a1, threads_per_block_1d>>>(d_dA1, d_dZ1, current_batch_size * HIDDEN_DIM); // d_dA1 is now dL/dZ1_biased

            // Gradient for W1: dW1 = X_batch^T * dZ1_biased_grad (which is now in d_dA1)
            dim3 blocks_transpose_X((current_batch_size + threads_per_block_2d.x-1)/threads_per_block_2d.x,
                                    (INPUT_DIM + threads_per_block_2d.y-1)/threads_per_block_2d.y);
            transpose_kernel<<<blocks_transpose_X, threads_per_block_2d>>>(d_X_batch, d_X_batch_T, current_batch_size, INPUT_DIM);
            
            dim3 blocks_matmul_dw1((HIDDEN_DIM + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                                   (INPUT_DIM + threads_per_block_2d.y - 1) / threads_per_block_2d.y);
            matmul_kernel<<<blocks_matmul_dw1, threads_per_block_2d>>>(d_X_batch_T, d_dA1, d_grad_W1, INPUT_DIM, HIDDEN_DIM, current_batch_size); // d_dA1 has dL/dZ1_biased

            // Gradient for B1: sum dZ1_biased_grad (now in d_dA1) over batch
            std::vector<double> h_dA1_vec(current_batch_size * HIDDEN_DIM); // Reusing h_dA1_vec
            HIP_CHECK(hipMemcpy(h_dA1_vec.data(), d_dA1, current_batch_size * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToHost));
            std::vector<double> h_grad_B1_vec(HIDDEN_DIM, 0.0);
            for(size_t i=0; i < current_batch_size; ++i) {
                for(int j=0; j < HIDDEN_DIM; ++j) {
                    h_grad_B1_vec[j] += h_dA1_vec[i * HIDDEN_DIM + j];
                }
            }
            HIP_CHECK(hipMemcpy(d_grad_B1, h_grad_B1_vec.data(), HIDDEN_DIM * sizeof(double), hipMemcpyHostToDevice));

            // --- Update Weights ---
            sgd_update<<<blocks_relu_a1, threads_per_block_1d>>>(d_W1, d_grad_W1, LEARNING_RATE, INPUT_DIM * HIDDEN_DIM);
            sgd_update<<<dim3((HIDDEN_DIM + threads_per_block_1d.x -1)/threads_per_block_1d.x), threads_per_block_1d>>>
                (d_B1, d_grad_B1, LEARNING_RATE, HIDDEN_DIM);
            sgd_update<<<dim3((HIDDEN_DIM * OUTPUT_DIM + threads_per_block_1d.x -1)/threads_per_block_1d.x), threads_per_block_1d>>>
                (d_W2, d_grad_W2, LEARNING_RATE, HIDDEN_DIM * OUTPUT_DIM);
            sgd_update<<<dim3((OUTPUT_DIM + threads_per_block_1d.x -1)/threads_per_block_1d.x), threads_per_block_1d>>>
                (d_B2, d_grad_B2, LEARNING_RATE, OUTPUT_DIM);
            
            HIP_CHECK(hipDeviceSynchronize()); // Ensure kernels complete before next batch / loss calculation

            // Calculate batch loss (on CPU for simplicity)
            std::vector<double> h_Y_pred_vec(current_batch_size * OUTPUT_DIM);
            HIP_CHECK(hipMemcpy(h_Y_pred_vec.data(), d_Y_pred, current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
            double batch_mse = 0.0;
            for (size_t i = 0; i < current_batch_size; ++i) {
                double diff = h_Y_pred_vec[i*OUTPUT_DIM] - h_Y_batch_vec[i*OUTPUT_DIM]; // Assuming OUTPUT_DIM = 1
                batch_mse += diff * diff;
            }
            epoch_loss += batch_mse;
        } // End batch loop

        epoch_loss /= train_size; // Average MSE for the epoch
        auto epoch_end_time = std::chrono::high_resolution_clock::now();
        auto epoch_duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end_time - epoch_start_time);
        std::cout << "[Epoch " << epoch + 1 << "/" << EPOCHS << "] Loss: " << epoch_loss << ", Time: " << epoch_duration.count() << " ms" << std::endl;
    } // End epoch loop


    // --- 5. Inference (Testing) ---
    std::cout << "\n[INFO] Starting Inference on Test Set..." << std::endl;
    std::vector<double> predictions_normalized;
    double total_test_mse = 0.0;
    int num_test_batches = (test_size + BATCH_SIZE - 1) / BATCH_SIZE;

    // Collect all predictions and actual values for variance calculation
    std::vector<double> all_predictions;
    std::vector<double> all_targets;

    for (int batch_idx = 0; batch_idx < num_test_batches; ++batch_idx) {
        size_t current_batch_offset = batch_idx * BATCH_SIZE;
        size_t current_batch_size = std::min((size_t)BATCH_SIZE, test_size - current_batch_offset);
         if (current_batch_size == 0) continue;

        std::vector<double> h_X_test_batch_vec(current_batch_size * INPUT_DIM);
        std::vector<double> h_Y_test_batch_vec(current_batch_size * OUTPUT_DIM); // For calculating MSE

        for(size_t i=0; i<current_batch_size; ++i) {
            for(int j=0; j<INPUT_DIM; ++j) {
                h_X_test_batch_vec[i * INPUT_DIM + j] = X_test[(current_batch_offset + i) * INPUT_DIM + j];
            }
             for(int j=0; j<OUTPUT_DIM; ++j) {
                h_Y_test_batch_vec[i * OUTPUT_DIM + j] = y_test[(current_batch_offset + i) * OUTPUT_DIM + j];
            }
        }

        HIP_CHECK(hipMemcpy(d_X_batch, h_X_test_batch_vec.data(), current_batch_size * INPUT_DIM * sizeof(double), hipMemcpyHostToDevice));

        // Forward Pass (same as training, without backward/update)
        dim3 blocks_matmul_z1((HIDDEN_DIM + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                              (current_batch_size + threads_per_block_2d.y - 1) / threads_per_block_2d.y);
        matmul_kernel<<<blocks_matmul_z1, threads_per_block_2d>>>(d_X_batch, d_W1, d_Z1, current_batch_size, HIDDEN_DIM, INPUT_DIM);
        HIP_CHECK(hipMemcpy(d_A1, d_Z1, current_batch_size * HIDDEN_DIM * sizeof(double), hipMemcpyDeviceToDevice));
        add_bias_kernel<<<blocks_matmul_z1, threads_per_block_2d>>>(d_A1, d_B1, current_batch_size, HIDDEN_DIM);
        dim3 blocks_relu_a1((current_batch_size * HIDDEN_DIM + threads_per_block_1d.x - 1) / threads_per_block_1d.x);
        relu_forward_kernel<<<blocks_relu_a1, threads_per_block_1d>>>(d_A1, current_batch_size * HIDDEN_DIM);
        dim3 blocks_matmul_z2((OUTPUT_DIM + threads_per_block_2d.x - 1) / threads_per_block_2d.x,
                              (current_batch_size + threads_per_block_2d.y - 1) / threads_per_block_2d.y);
        matmul_kernel<<<blocks_matmul_z2, threads_per_block_2d>>>(d_A1, d_W2, d_Z2, current_batch_size, OUTPUT_DIM, HIDDEN_DIM);
        HIP_CHECK(hipMemcpy(d_Y_pred, d_Z2, current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToDevice));
        add_bias_kernel<<<blocks_matmul_z2, threads_per_block_2d>>>(d_Y_pred, d_B2, current_batch_size, OUTPUT_DIM);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<double> h_Y_pred_batch_vec(current_batch_size * OUTPUT_DIM);
        HIP_CHECK(hipMemcpy(h_Y_pred_batch_vec.data(), d_Y_pred, current_batch_size * OUTPUT_DIM * sizeof(double), hipMemcpyDeviceToHost));
        
        for(size_t i=0; i < current_batch_size; ++i) {
            predictions_normalized.push_back(h_Y_pred_batch_vec[i*OUTPUT_DIM]); // Assuming OUTPUT_DIM = 1
            all_predictions.push_back(h_Y_pred_batch_vec[i*OUTPUT_DIM]);
            all_targets.push_back(h_Y_test_batch_vec[i*OUTPUT_DIM]);

            double diff = h_Y_pred_batch_vec[i*OUTPUT_DIM] - h_Y_test_batch_vec[i*OUTPUT_DIM];
            total_test_mse += diff * diff;
        }
    }
    total_test_mse /= test_size;
    std::cout << "[INFO] Test MSE (normalized): " << total_test_mse << std::endl;

    // --- Calculate Variance ---
    // 1. Calculate the mean of predictions
    double sum_predictions = std::accumulate(all_predictions.begin(), all_predictions.end(), 0.0);
    double mean_prediction = sum_predictions / all_predictions.size();

    // 2. Calculate the variance
    double sq_sum = std::inner_product(all_predictions.begin(), all_predictions.end(), all_predictions.begin(), 0.0);
    double variance_predictions = sq_sum / all_predictions.size() - mean_prediction * mean_prediction;

    std::cout << "[INFO] Variance of predictions (normalized): " << variance_predictions << std::endl;

    // --- Calculate Variance of targets ---
    // 1. Calculate the mean of targets
    double sum_targets = std::accumulate(all_targets.begin(), all_targets.end(), 0.0);
    double mean_target = sum_targets / all_targets.size();

    // 2. Calculate the variance
    double sq_sum_targets = std::inner_product(all_targets.begin(), all_targets.end(), all_targets.begin(), 0.0);
    double variance_targets = sq_sum_targets / all_targets.size() - mean_target * mean_target;

    std::cout << "[INFO] Variance of targets (normalized): " << variance_targets << std::endl;

    // --- Calculate Variance of errors ---
    std::vector<double> errors;
    std::transform(all_predictions.begin(), all_predictions.end(), all_targets.begin(), std::back_inserter(errors),
                   [](double pred, double target){ return pred - target; });

    double sum_errors = std::accumulate(errors.begin(), errors.end(), 0.0);
    double mean_error = sum_errors / errors.size();

    double sq_sum_errors = std::inner_product(errors.begin(), errors.end(), errors.begin(), 0.0);
    double variance_errors = sq_sum_errors / errors.size() - mean_error * mean_error;

    std::cout << "[INFO] Variance of errors (normalized): " << variance_errors << std::endl;

    std::vector<double> predictions_denormalized = predictions_normalized;
    denormalize_data(predictions_denormalized, min_bw, max_bw);
    
    std::vector<double> y_test_denormalized = y_test; // y_test is already a portion of normalized_data
    denormalize_data(y_test_denormalized, min_bw, max_bw);


    double total_test_mse_denormalized = 0.0;
     for (size_t i = 0; i < test_size; ++i) {
        double diff = predictions_denormalized[i] - y_test_denormalized[i];
        total_test_mse_denormalized += diff * diff;
    }
    total_test_mse_denormalized /= test_size;
    std::cout << "[INFO] Test MSE (denormalized): " << total_test_mse_denormalized << std::endl;


    std::cout << "\n[INFO] Sample Predictions (denormalized):" << std::endl;
    for (size_t i = 0; i < std::min((size_t)10, test_size); ++i) {
        std::cout << "Predicted: " << predictions_denormalized[i] << ", Actual: " << y_test_denormalized[i] << std::endl;
    }

    // --- 6. Free Device Memory ---
    hipFree(d_W1); hipFree(d_B1); hipFree(d_W2); hipFree(d_B2);
    hipFree(d_X_batch); hipFree(d_Y_batch); hipFree(d_Y_pred);
    hipFree(d_Z1); hipFree(d_A1); hipFree(d_Z2);
    hipFree(d_grad_W1); hipFree(d_grad_B1); hipFree(d_grad_W2); hipFree(d_grad_B2);
    hipFree(d_dZ2); hipFree(d_dA1); hipFree(d_dZ1);
    hipFree(d_A1_T); hipFree(d_X_batch_T); hipFree(d_W2_T);
    hipFree(d_mse_elements);

    auto global_end_time = std::chrono::high_resolution_clock::now();
    auto global_duration = std::chrono::duration_cast<std::chrono::seconds>(global_end_time - global_start_time);
    std::cout << "\n[INFO] Total execution time: " << global_duration.count() << " seconds." << std::endl;
    std::cout << "[INFO] MLP training and inference complete." << std::endl;
    return 0;
}