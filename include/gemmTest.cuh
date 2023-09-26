#ifndef GPU_GEMM_TEST__
#define GPU_GEMM_TEST__

#include <DNGemm.cuh>


double calc_Performance(const float *h_a, const float *h_b, 
                        const int size_a, const int size_b, const int size_c,
                        const int M, const int N, const int K, 
                        const int repeat, executeType exectp, float &avg_runtime);

void check_gemm(const float *h_a, const float *h_b, const float *h_check,
                const int size_a, const int size_b, const int size_c,
                const int M, const int K, const int N);


void CublasSgemm(const float *h_a, const float *h_b, const float *h_check,
                 const int size_a, const int size_b, const int size_c,
                 const int M, const int K, const int N);

void check_acc(const float *h_a, const float *h_b,
               const int size_a, const int size_b, const int size_c,
               const int M, const int K, const int N);

void get_Performance(const float *h_a, const float *h_b,
               const int size_a, const int size_b, const int size_c,
               const int M, const int K, const int N, const int repeat);

void test_diff_size_gemm();


void test_diff_size_gemm_acc();

void test_diff_size_gemm_ones();

#endif