#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono> // For timing
#include <iomanip> // For std::fixed and std::setprecision

// 编译
// hipcc lesson1_sourcefile_dcu.cpp -o outputfile_dcu
// 执行
// ./outputfile_dcu

#define N_DIM 1024 
#define M_DIM 2048
#define P_DIM 512

// HIP error checking macro
#define HIP_CHECK(error) \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: %s (%d) at %s:%d\n", hipGetErrorString(error), error, __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// 主要修改函数
__global__ void matmul_kernel(const double* A, const double* B, double* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        double sum = 0.0;
        for (int k = 0; k < m; ++k) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

void init_matrix(std::vector<double>& mat, int rows, int cols) {
    mat.resize(rows * cols); // Ensure correct size
    std::mt19937 gen(42); // Fixed seed for reproducibility
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (auto& x : mat)
        x = dist(gen);
}

void matmul_cpu(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& C, int n, int m, int p) {
    if(C.size() != (size_t)n * p) C.resize(n*p); // Ensure C is correctly sized
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j) {
            double sum = 0.0;
            for (int k = 0; k < m; ++k)
                sum += A[i * m + k] * B[k * p + j];
            C[i * p + j] = sum;
        }
}

bool validate(const std::vector<double>& ref, const std::vector<double>& test, int n, int p, double tol = 1e-6) {
    if (ref.size() != test.size() || ref.size() != (size_t)n * p) {
        std::cerr << "Validation error: Size mismatch. ref: " << ref.size() << ", test: " << test.size() << ", expected: " << (size_t)n*p << std::endl;
        return false;
    }
    for (size_t i = 0; i < ref.size(); ++i) {
        if (std::abs(ref[i] - test[i]) > tol) {
            std::cerr << "Validation failed at index " << i << ": ref=" << ref[i] << ", test=" << test[i] << ", diff=" << std::abs(ref[i] - test[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    const int n_val = N_DIM;
    const int m_val = M_DIM;
    const int p_val = P_DIM;
    const double total_flops = 2.0 * n_val * m_val * p_val;

    std::cout << "Matrix dimensions: N=" << n_val << ", M=" << m_val << ", P=" << p_val << std::endl;
    std::cout << "Total FLOPs for matmul: " << total_flops << std::endl;

    std::vector<double> h_A, h_B, h_C_gpu(n_val * p_val), h_C_cpu(n_val * p_val);
    init_matrix(h_A, n_val, m_val);
    init_matrix(h_B, m_val, p_val);

    // CPU baseline
    std::cout << "\nRunning CPU baseline..." << std::endl;
    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    matmul_cpu(h_A, h_B, h_C_cpu, n_val, m_val, p_val);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cpu_duration_ms = cpu_end_time - cpu_start_time;
    double cpu_duration_s = cpu_duration_ms.count() / 1000.0;
    double cpu_gflops = (cpu_duration_s > 0) ? (total_flops / (cpu_duration_s * 1e9)) : 0;
    std::cout << "[CPU] Time: " << cpu_duration_ms.count() << " ms" << std::endl;
    std::cout << "[CPU] GFLOPS: " << std::fixed << std::setprecision(2) << cpu_gflops << std::endl;
    std::cout << "[CPU] Baseline computation complete." << std::endl;


    // 主要修改部分
    std::cout << "\nRunning HIP GPU computation..." << std::endl;
    double *d_A, *d_B, *d_C;
    size_t size_A = (size_t)n_val * m_val * sizeof(double);
    size_t size_B = (size_t)m_val * p_val * sizeof(double);
    size_t size_C = (size_t)n_val * p_val * sizeof(double);

    // 分配内存
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    // 转移数据
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

    
    dim3 threadsPerBlock(16, 16); 
    dim3 numBlocks((p_val + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n_val + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Use hipEvent_t for accurate GPU timing
    hipEvent_t start_event, stop_event;
    HIP_CHECK(hipEventCreate(&start_event));
    HIP_CHECK(hipEventCreate(&stop_event));

    // Record start event
    HIP_CHECK(hipEventRecord(start_event, 0));

    hipLaunchKernelGGL(matmul_kernel, numBlocks, threadsPerBlock, 0, 0, d_A, d_B, d_C, n_val, m_val, p_val);
    HIP_CHECK(hipGetLastError()); // Check for kernel launch errors

    // Record stop event and synchronize
    HIP_CHECK(hipEventRecord(stop_event, 0));
    HIP_CHECK(hipEventSynchronize(stop_event)); // Wait for the event to complete

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start_event, stop_event));
    
    double gpu_duration_s = milliseconds / 1000.0;
    double gpu_gflops = (gpu_duration_s > 0) ? (total_flops / (gpu_duration_s * 1e9)) : 0;

    std::cout << "[HIP] Kernel Time: " << milliseconds << " ms" << std::endl;
    std::cout << "[HIP] GFLOPS: " << std::fixed << std::setprecision(2) << gpu_gflops << std::endl;


    // Copy result from device to host
    HIP_CHECK(hipMemcpy(h_C_gpu.data(), d_C, size_C, hipMemcpyDeviceToHost));

    // Validate
    if (validate(h_C_cpu, h_C_gpu, n_val, p_val)) {
        std::cout << "[HIP] Valid: 1 (Results match CPU)" << std::endl;
    } else {
        std::cout << "[HIP] Valid: 0 (Results DO NOT match CPU)" << std::endl;
    }

    // Free device memory
    HIP_CHECK(hipEventDestroy(start_event));
    HIP_CHECK(hipEventDestroy(stop_event));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    std::cout << "\nFinished." << std::endl;
    return 0;
}