#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <mpi.h>
#include <chrono>
#include <iomanip> // For std::fixed and std::setprecision

// 编译执行方式参考：
// 编译，也可以使用g++，但使用MPI时需使用mpic
// mpic++ -fopenmp -o outputfile sourcefile.cpp
//
// 运行 (如果包含MPI，则使用mpirun):
// mpirun -np <number_of_processes> ./outputfile
// 例如，使用4个进程:
// mpirun -np 4 ./outputfile
// 如果只想测试非MPI部分（不推荐，因为代码结构现在是为MPI设计的）：
// ./outputfile (但这会因为MPI_Init未正确调用而可能出错或行为不当)

// 初始化矩阵（以一维数组形式表示），用于随机填充浮点数
void init_matrix(std::vector<double>& mat, int rows, int cols) {
    std::mt19937 gen(42); // 使用固定的种子以便结果可复现
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    for (int i = 0; i < rows * cols; ++i)
        mat[i] = dist(gen);
}

// 验证计算优化后的矩阵计算和baseline实现是否结果一致
bool validate(const std::vector<double>& A, const std::vector<double>& B, int rows, int cols, double tol = 1e-6) {
    if (A.size() != B.size() || A.size() != (size_t)rows * cols) {
        std::cerr << "Validation error: Matrix dimensions mismatch. A.size()=" << A.size() 
                  << ", B.size()=" << B.size() << ", rows*cols=" << (size_t)rows*cols << std::endl;
        return false;
    }
    for (int i = 0; i < rows * cols; ++i)
        if (std::abs(A[i] - B[i]) > tol) {
            // std::cerr << "Validation failed at index " << i << ": A=" << A[i] << ", B=" << B[i] << std::endl;
            return false;
        }
    return true;
}

// 基础的矩阵乘法baseline实现（使用一维数组）
void matmul_baseline(const std::vector<double>& A,
                     const std::vector<double>& B,
                     std::vector<double>& C, int N, int M, int P) {
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < P; ++j) {
            C[i * P + j] = 0; // 初始化 C 的元素
            for (int k = 0; k < M; ++k)
                C[i * P + j] += A[i * M + k] * B[k * P + j];
        }
}

// 方式1：利用OpenMP进行多线程并发的编程
void matmul_openmp(const std::vector<double>& A,
                   const std::vector<double>& B,
                   std::vector<double>& C, int N, int M, int P) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0; 
            for (int k = 0; k < M; ++k) {
                sum += A[i * M + k] * B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }
}

// 方式2：利用子块并行思想，进行缓存友好型的并行优化方法
void matmul_block_tiling(const std::vector<double>& A,
                         const std::vector<double>& B,
                         std::vector<double>& C, int N, int M, int P, int block_size = 64) {
    for (int i0 = 0; i0 < N; i0 += block_size) {
        for (int j0 = 0; j0 < P; j0 += block_size) {
            for (int k0 = 0; k0 < M; k0 += block_size) {
                for (int i = i0; i < std::min(i0 + block_size, N); ++i) {
                    for (int j = j0; j < std::min(j0 + block_size, P); ++j) {
                        for (int k = k0; k < std::min(k0 + block_size, M); ++k) {
                            C[i * P + j] += A[i * M + k] * B[k * P + j];
                        }
                    }
                }
            }
        }
    }
}

// 方式3：利用MPI消息传递，实现多进程并行优化
// C_ref_main is passed for rank 0 to optionally validate against it.
void matmul_mpi(int N, int M, int P, const std::vector<double>& C_ref_main_for_validation) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    std::chrono::high_resolution_clock::time_point mpi_start_time, mpi_end_time;

    if (rank == 0) {
        mpi_start_time = std::chrono::high_resolution_clock::now();
    }

    int rows_per_proc_base = N / size;
    int remainder_rows = N % size;

    int local_N = rows_per_proc_base + (rank < remainder_rows ? 1 : 0);
    
    std::vector<double> local_A(local_N * M);
    std::vector<double> B_all(M * P); 
    std::vector<double> local_C(local_N * P, 0.0);

    std::vector<double> A_all_mpi;    
    std::vector<double> C_result_mpi_gathered; 

    std::vector<int> sendcounts_A(size); 
    std::vector<int> displs_A(size);     
    std::vector<int> recvcounts_C(size); 
    std::vector<int> displs_C(size);     

    if (rank == 0) {
        A_all_mpi.resize(N * M);
        C_result_mpi_gathered.resize(N * P);

        // Initialize A and B on rank 0. Since init_matrix uses a fixed seed,
        // A_all_mpi and B_all here will be consistent with A and B in main (rank 0).
        init_matrix(A_all_mpi, N, M); 
        init_matrix(B_all, M, P); 

        int current_displ_A_val = 0;
        int current_displ_C_val = 0;
        for (int i = 0; i < size; ++i) {
            int num_rows_for_proc = rows_per_proc_base + (i < remainder_rows ? 1 : 0);
            
            sendcounts_A[i] = num_rows_for_proc * M;
            displs_A[i] = current_displ_A_val;
            current_displ_A_val += sendcounts_A[i];

            recvcounts_C[i] = num_rows_for_proc * P;
            displs_C[i] = current_displ_C_val;
            current_displ_C_val += recvcounts_C[i];
        }
    }

    MPI_Bcast(B_all.data(), M * P, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(A_all_mpi.data(), sendcounts_A.data(), displs_A.data(), MPI_DOUBLE,
                 local_A.data(), local_N * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < local_N; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < M; ++k) {
                sum += local_A[i * M + k] * B_all[k * P + j];
            }
            local_C[i * P + j] = sum;
        }
    }

    MPI_Gatherv(local_C.data(), local_N * P, MPI_DOUBLE,
                C_result_mpi_gathered.data(), recvcounts_C.data(), displs_C.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        mpi_end_time = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration_ms_chrono = mpi_end_time - mpi_start_time;
        double duration_s = duration_ms_chrono.count() / 1000.0;
        double gflops = (2.0 * N * M * P) / (duration_s * 1e9);

        std::cout << "[MPI] Computation complete on root." << std::endl;
        std::cout << "[MPI] Time: " << duration_ms_chrono.count() << " ms" << std::endl;
        std::cout << "[MPI] GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
        
        bool is_valid = validate(C_result_mpi_gathered, C_ref_main_for_validation, N, P);
        std::cout << "[MPI] Valid: " << is_valid << std::endl;
    }
}

// 方式4：其他方式（ikj loop order）
void matmul_other(const std::vector<double>& A,
                  const std::vector<double>& B,
                  std::vector<double>& C, int N, int M, int P) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < M; ++k) {
            double val_A = A[i * M + k];
            for (int j = 0; j < P; ++j) {
                C[i * P + j] += val_A * B[k * P + j];
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int mpi_rank, mpi_world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_world_size);

    const int N = 4096, M = 4096, P = 4096; 
    const double total_flops = 2.0 * N * M * P;

    std::vector<double> A(N * M);
    std::vector<double> B(M * P);
    std::vector<double> C(N * P, 0.0); 
    std::vector<double> C_ref(N * P, 0.0);

    // Initialization and non-MPI tests are done only by rank 0
    if (mpi_rank == 0) {
        std::cout << "Running tests on MPI Rank 0. MPI World Size: " << mpi_world_size << std::endl;
        std::cout << "Matrix dimensions: N=" << N << ", M=" << M << ", P=" << P << std::endl;
        std::cout << "Total FLOPs for matmul: " << total_flops << std::endl << std::endl;


        init_matrix(A, N, M);
        init_matrix(B, M, P);

        std::chrono::high_resolution_clock::time_point start_time, end_time;
        std::chrono::duration<double, std::milli> duration_ms_chrono;
        double duration_s;
        double gflops;

        // --- Baseline ---
        std::cout << "Running Baseline..." << std::endl;
        start_time = std::chrono::high_resolution_clock::now();
        matmul_baseline(A, B, C_ref, N, M, P); // C_ref is the reference
        end_time = std::chrono::high_resolution_clock::now();
        duration_ms_chrono = end_time - start_time;
        duration_s = duration_ms_chrono.count() / 1000.0;
        gflops = total_flops / (duration_s * 1e9);
        std::cout << "[Baseline] Time: " << duration_ms_chrono.count() << " ms" << std::endl;
        std::cout << "[Baseline] GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
        std::cout << "[Baseline] Reference computation complete.\n" << std::endl;

        // --- OpenMP ---
        std::cout << "Running OpenMP..." << std::endl;
        std::fill(C.begin(), C.end(), 0.0); 
        start_time = std::chrono::high_resolution_clock::now();
        matmul_openmp(A, B, C, N, M, P);
        end_time = std::chrono::high_resolution_clock::now();
        duration_ms_chrono = end_time - start_time;
        duration_s = duration_ms_chrono.count() / 1000.0;
        gflops = total_flops / (duration_s * 1e9);
        std::cout << "[OpenMP] Time: " << duration_ms_chrono.count() << " ms" << std::endl;
        std::cout << "[OpenMP] GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
        std::cout << "[OpenMP] Valid: " << validate(C, C_ref, N, P) << std::endl << std::endl;

        // --- Block Tiling ---
        std::cout << "Running Block Tiling..." << std::endl;
        std::fill(C.begin(), C.end(), 0.0); 
        start_time = std::chrono::high_resolution_clock::now();
        matmul_block_tiling(A, B, C, N, M, P);
        end_time = std::chrono::high_resolution_clock::now();
        duration_ms_chrono = end_time - start_time;
        duration_s = duration_ms_chrono.count() / 1000.0;
        gflops = total_flops / (duration_s * 1e9);
        std::cout << "[Block Tiling] Time: " << duration_ms_chrono.count() << " ms" << std::endl;
        std::cout << "[Block Tiling] GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
        std::cout << "[Block Tiling] Valid: " << validate(C, C_ref, N, P) << std::endl << std::endl;

        // --- Other (ikj loop) ---
        std::cout << "Running Other (ikj loop)..." << std::endl;
        std::fill(C.begin(), C.end(), 0.0); 
        start_time = std::chrono::high_resolution_clock::now();
        matmul_other(A, B, C, N, M, P);
        end_time = std::chrono::high_resolution_clock::now();
        duration_ms_chrono = end_time - start_time;
        duration_s = duration_ms_chrono.count() / 1000.0;
        gflops = total_flops / (duration_s * 1e9);
        std::cout << "[Other] Time: " << duration_ms_chrono.count() << " ms" << std::endl;
        std::cout << "[Other] GFLOPS: " << std::fixed << std::setprecision(2) << gflops << std::endl;
        std::cout << "[Other] Valid: " << validate(C, C_ref, N, P) << std::endl << std::endl;
    }

    // Synchronize before MPI test to ensure C_ref is ready on rank 0 if it were broadcasted
    MPI_Barrier(MPI_COMM_WORLD); 

    // --- MPI Test ---
    if (mpi_rank == 0) {
         std::cout << "Running MPI..." << std::endl;
    }
    matmul_mpi(N, M, P, C_ref); // C_ref is only meaningfully used by rank 0 inside matmul_mpi

    MPI_Finalize();
    return 0;
}