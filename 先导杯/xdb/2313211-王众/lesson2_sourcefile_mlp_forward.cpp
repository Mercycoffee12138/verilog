#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib> // For rand, RAND_MAX
#include <cmath>   // For fmax, fabs
#include <cstdio>  // For std::cout if used for debugging
#include <chrono>  // For timing
#include <iomanip> // For std::fixed and std::setprecision
#include <numeric> // For std::iota (optional, for debugging)
#include <algorithm> // For std::fill

// 编译文件
// hipcc lesson2_sourcefile_mlp_forward.cpp -o mlp_forward
// 执行文件
// ./mlp_forward 或者 hipprof ./mlp_forward


#define BATCH 1024
#define I 10
#define H 20
#define O 5

#define TILE_WIDTH 16 // For tiled matrix multiplication

// --- GPU Kernels ---

// Basic Matrix Multiplication Kernel
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

// Tiled Matrix Multiplication Kernel (Optimized)
__global__ void matmul_kernel_tiled(const double* A, const double* B, double* C, int M, int N, int K) {
    __shared__ double tile_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ double tile_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Calculate the row and col of the C element to work on
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    double sum = 0.0;

    // Loop over the tiles of A and B required to compute the C element
    for (int t = 0; t < (K + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {
        // Load a tile of A into shared memory
        if (row < M && (t * TILE_WIDTH + tx) < K) {
            tile_A[ty][tx] = A[row * K + (t * TILE_WIDTH + tx)];
        } else {
            tile_A[ty][tx] = 0.0;
        }

        // Load a tile of B into shared memory
        if (col < N && (t * TILE_WIDTH + ty) < K) {
            tile_B[ty][tx] = B[(t * TILE_WIDTH + ty) * N + col];
        } else {
            tile_B[ty][tx] = 0.0;
        }
        __syncthreads(); // Ensure all threads in the block have loaded their data

        // Multiply the two tiles
        for (int i = 0; i < TILE_WIDTH; ++i) {
            sum += tile_A[ty][i] * tile_B[i][tx];
        }
        __syncthreads(); // Ensure all threads in the block have finished using the current tiles
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// Add Bias Kernel
__global__ void add_bias_kernel(double* matrix, const double* bias, int M_rows, int N_cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M_rows && col < N_cols) {
        matrix[row * N_cols + col] += bias[col];
    }
}

// ReLU Kernel
__global__ void relu_kernel(double* A, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        A[idx] = fmax(0.0, A[idx]);
    }
}

// --- CPU Functions for Validation ---
void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int M, int N, int K) {
    std::fill(C.begin(), C.end(), 0.0);
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            double sum = 0.0;
            for (int k_idx = 0; k_idx < K; ++k_idx) {
                sum += A[r * K + k_idx] * B[k_idx * N + c];
            }
            C[r * N + c] = sum;
        }
    }
}

void add_bias_cpu(std::vector<double>& matrix, const std::vector<double>& bias, int M_rows, int N_cols) {
    for (int r = 0; r < M_rows; ++r) {
        for (int c = 0; c < N_cols; ++c) {
            matrix[r * N_cols + c] += bias[c];
        }
    }
}

void relu_cpu(std::vector<double>& A) {
    for (size_t i = 0; i < A.size(); ++i) {
        A[i] = fmax(0.0, A[i]);
    }
}

// --- Utility Functions ---
void random_init(std::vector<double>& mat) {
    for (auto& val : mat) {
        val = static_cast<double>(rand()) / RAND_MAX * 2.0 - 1.0;
    }
}

bool validate_results(const std::vector<double>& gpu_res, const std::vector<double>& cpu_res, double tolerance = 1e-5) {
    if (gpu_res.size() != cpu_res.size()) {
        std::cerr << "Validation Error: Size mismatch! GPU: " << gpu_res.size() << ", CPU: " << cpu_res.size() << std::endl;
        return false;
    }
    for (size_t i = 0; i < gpu_res.size(); ++i) {
        if (std::fabs(gpu_res[i] - cpu_res[i]) > tolerance) {
            std::cerr << "Validation Error at index " << i << ": GPU=" << gpu_res[i] << ", CPU=" << cpu_res[i]
                      << ", Diff=" << std::fabs(gpu_res[i] - cpu_res[i]) << std::endl;
            return false;
        }
    }
    return true;
}

void print_mlp_config() {
    std::cout << "MLP Configuration:" << std::endl;
    std::cout << "Batch Size (BATCH): " << BATCH << std::endl;
    std::cout << "Input Dim (I):    " << I << std::endl;
    std::cout << "Hidden Dim (H):   " << H << std::endl;
    std::cout << "Output Dim (O):   " << O << std::endl;
    std::cout << "------------------------------------" << std::endl;
}

double calculate_total_flops() {
    double flops_matmul1 = 2.0 * BATCH * H * I;
    double flops_add_bias1 = static_cast<double>(BATCH * H);
    double flops_relu = static_cast<double>(BATCH * H); // Counting ReLU as one FLOP per element
    double flops_matmul2 = 2.0 * BATCH * O * H;
    double flops_add_bias2 = static_cast<double>(BATCH * O);
    return flops_matmul1 + flops_add_bias1 + flops_relu + flops_matmul2 + flops_add_bias2;
}


int main() {
    std::vector<double> h_X(BATCH * I), h_W1(I * H), h_B1(H), h_W2(H * O), h_B2(O);
    std::vector<double> h_H_cpu(BATCH * H), h_Y_cpu(BATCH * O);
    std::vector<double> h_Y_gpu(BATCH * O);

    srand(0);
    random_init(h_X);
    random_init(h_W1);
    random_init(h_B1);
    random_init(h_W2);
    random_init(h_B2);

    print_mlp_config();
    double total_flops = calculate_total_flops();
    std::cout << "Total Theoretical FLOPs: " << std::fixed << std::setprecision(0) << total_flops << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // --- CPU MLP Forward Pass (for validation) ---
    std::cout << "Running CPU MLP Forward Pass for reference..." << std::endl;
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    // Hidden layer: H_cpu = ReLU(X * W1 + B1)
    matmul_cpu(h_X, h_W1, h_H_cpu, BATCH, H, I);
    add_bias_cpu(h_H_cpu, h_B1, BATCH, H);
    relu_cpu(h_H_cpu);
    // Output layer: Y_cpu = H_cpu * W2 + B2
    matmul_cpu(h_H_cpu, h_W2, h_Y_cpu, BATCH, O, H);
    add_bias_cpu(h_Y_cpu, h_B2, BATCH, O);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration_ms = cpu_end_time - cpu_start_time;
    std::cout << "CPU Execution Time: " << cpu_duration_ms.count() << " ms" << std::endl;
    std::cout << "------------------------------------" << std::endl;


    // --- GPU Setup ---
    double *d_X, *d_W1, *d_B1, *d_H_gpu, *d_W2, *d_B2, *d_Y_gpu;
    size_t size_X = BATCH * I * sizeof(double);
    size_t size_W1 = I * H * sizeof(double);
    size_t size_B1 = H * sizeof(double);
    size_t size_d_H_gpu = BATCH * H * sizeof(double);
    size_t size_W2 = H * O * sizeof(double);
    size_t size_B2 = O * sizeof(double);
    size_t size_d_Y_gpu = BATCH * O * sizeof(double);

    hipMalloc((void**)&d_X, size_X);
    hipMalloc((void**)&d_W1, size_W1);
    hipMalloc((void**)&d_B1, size_B1);
    hipMalloc((void**)&d_H_gpu, size_d_H_gpu);
    hipMalloc((void**)&d_W2, size_W2);
    hipMalloc((void**)&d_B2, size_B2);
    hipMalloc((void**)&d_Y_gpu, size_d_Y_gpu);

    hipMemcpy(d_X, h_X.data(), size_X, hipMemcpyHostToDevice);
    hipMemcpy(d_W1, h_W1.data(), size_W1, hipMemcpyHostToDevice);
    hipMemcpy(d_B1, h_B1.data(), size_B1, hipMemcpyHostToDevice);
    hipMemcpy(d_W2, h_W2.data(), size_W2, hipMemcpyHostToDevice);
    hipMemcpy(d_B2, h_B2.data(), size_B2, hipMemcpyHostToDevice);

    const int TPB_2D_BASIC = 16; // Threads per block for basic matmul
    const int TPB_1D_RELU = 256;


    // --- GPU MLP Forward Pass (Basic matmul_kernel) ---
    std::cout << "Running GPU MLP (Basic matmul_kernel)..." << std::endl;
    hipDeviceSynchronize();
    auto gpu_basic_start_time = std::chrono::high_resolution_clock::now();

    dim3 threads_matmul1_basic(TPB_2D_BASIC, TPB_2D_BASIC);
    dim3 blocks_matmul1_basic((H + TPB_2D_BASIC - 1) / TPB_2D_BASIC, (BATCH + TPB_2D_BASIC - 1) / TPB_2D_BASIC);
    matmul_kernel<<<blocks_matmul1_basic, threads_matmul1_basic>>>(d_X, d_W1, d_H_gpu, BATCH, H, I);

    dim3 threads_bias1_basic(TPB_2D_BASIC, TPB_2D_BASIC); // Can reuse TPB_2D_BASIC
    dim3 blocks_bias1_basic((H + TPB_2D_BASIC - 1) / TPB_2D_BASIC, (BATCH + TPB_2D_BASIC - 1) / TPB_2D_BASIC);
    add_bias_kernel<<<blocks_bias1_basic, threads_bias1_basic>>>(d_H_gpu, d_B1, BATCH, H);

    dim3 threads_relu1(TPB_1D_RELU);
    dim3 blocks_relu1((BATCH * H + TPB_1D_RELU - 1) / TPB_1D_RELU);
    relu_kernel<<<blocks_relu1, threads_relu1>>>(d_H_gpu, BATCH * H);

    dim3 threads_matmul2_basic(TPB_2D_BASIC, TPB_2D_BASIC);
    dim3 blocks_matmul2_basic((O + TPB_2D_BASIC - 1) / TPB_2D_BASIC, (BATCH + TPB_2D_BASIC - 1) / TPB_2D_BASIC);
    matmul_kernel<<<blocks_matmul2_basic, threads_matmul2_basic>>>(d_H_gpu, d_W2, d_Y_gpu, BATCH, O, H);

    dim3 threads_bias2_basic(TPB_2D_BASIC, TPB_2D_BASIC);
    dim3 blocks_bias2_basic((O + TPB_2D_BASIC - 1) / TPB_2D_BASIC, (BATCH + TPB_2D_BASIC - 1) / TPB_2D_BASIC);
    add_bias_kernel<<<blocks_bias2_basic, threads_bias2_basic>>>(d_Y_gpu, d_B2, BATCH, O);
    
    hipDeviceSynchronize();
    auto gpu_basic_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_basic_duration_ms = gpu_basic_end_time - gpu_basic_start_time;
    
    hipMemcpy(h_Y_gpu.data(), d_Y_gpu, size_d_Y_gpu, hipMemcpyDeviceToHost);
    bool basic_valid = validate_results(h_Y_gpu, h_Y_cpu);
    double gflops_basic = (total_flops / (gpu_basic_duration_ms.count() / 1000.0)) / 1e9;

    std::cout << "GPU (Basic) Execution Time: " << gpu_basic_duration_ms.count() << " ms" << std::endl;
    std::cout << "GPU (Basic) GFLOPS: " << std::fixed << std::setprecision(2) << gflops_basic << std::endl;
    std::cout << "GPU (Basic) Validation: " << (basic_valid ? "PASSED" : "FAILED") << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // --- GPU MLP Forward Pass (Tiled matmul_kernel_tiled) ---
    std::cout << "Running GPU MLP (Tiled matmul_kernel_tiled)..." << std::endl;
    // Reset intermediate and output GPU buffers if necessary (though they are overwritten)
    // hipMemset(d_H_gpu, 0, size_d_H_gpu); // Optional: clear if needed
    // hipMemset(d_Y_gpu, 0, size_d_Y_gpu); // Optional: clear if needed

    hipDeviceSynchronize();
    auto gpu_tiled_start_time = std::chrono::high_resolution_clock::now();

    // For matmul_kernel_tiled, threads per block must match TILE_WIDTH
    dim3 threads_matmul_tiled(TILE_WIDTH, TILE_WIDTH);
    dim3 blocks_matmul1_tiled((H + TILE_WIDTH - 1) / TILE_WIDTH, (BATCH + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel_tiled<<<blocks_matmul1_tiled, threads_matmul_tiled>>>(d_X, d_W1, d_H_gpu, BATCH, H, I);

    // Bias and ReLU kernels remain the same
    add_bias_kernel<<<blocks_bias1_basic, threads_bias1_basic>>>(d_H_gpu, d_B1, BATCH, H); // Reusing basic launch params
    relu_kernel<<<blocks_relu1, threads_relu1>>>(d_H_gpu, BATCH * H);

    dim3 blocks_matmul2_tiled((O + TILE_WIDTH - 1) / TILE_WIDTH, (BATCH + TILE_WIDTH - 1) / TILE_WIDTH);
    matmul_kernel_tiled<<<blocks_matmul2_tiled, threads_matmul_tiled>>>(d_H_gpu, d_W2, d_Y_gpu, BATCH, O, H);
    
    add_bias_kernel<<<blocks_bias2_basic, threads_bias2_basic>>>(d_Y_gpu, d_B2, BATCH, O); // Reusing basic launch params

    hipDeviceSynchronize();
    auto gpu_tiled_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> gpu_tiled_duration_ms = gpu_tiled_end_time - gpu_tiled_start_time;

    hipMemcpy(h_Y_gpu.data(), d_Y_gpu, size_d_Y_gpu, hipMemcpyDeviceToHost);
    bool tiled_valid = validate_results(h_Y_gpu, h_Y_cpu);
    double gflops_tiled = (total_flops / (gpu_tiled_duration_ms.count() / 1000.0)) / 1e9;

    std::cout << "GPU (Tiled) Execution Time: " << gpu_tiled_duration_ms.count() << " ms" << std::endl;
    std::cout << "GPU (Tiled) GFLOPS: " << std::fixed << std::setprecision(2) << gflops_tiled << std::endl;
    std::cout << "GPU (Tiled) Validation: " << (tiled_valid ? "PASSED" : "FAILED") << std::endl;
    std::cout << "------------------------------------" << std::endl;

    // --- Cleanup ---
    hipFree(d_X);
    hipFree(d_W1);
    hipFree(d_B1);
    hipFree(d_H_gpu);
    hipFree(d_W2);
    hipFree(d_B2);
    hipFree(d_Y_gpu);

    return 0;
}