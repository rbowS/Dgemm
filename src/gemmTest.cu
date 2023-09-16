#include <gemmTest.cuh>



void cpuSgemm(
    const float *a, const float *b, float *c, const int M, const int N, const int K)
{

    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < N; n++)
        {
            float psum = 0.0;
            for (int k = 0; k < K; k++)
            {
                psum += a[m * K + k] * b[k * N + n];
            }
            c[m * N + n] = psum;
        }
    }
}


double calc_Performance(const float *h_a, const float *h_b, 
                        const int size_a, const int size_b, const int size_c,
                        const int M, const int N, const int K, 
                        const int repeat, executeType exectp, float &avg_runtime) 
{
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    GpuTimer timer;
    float time_cost;
    switch (exectp) {
        case EXECcuBLAS: {
            cublasHandle_t cublas_handle;
            cublasCreate(&cublas_handle);
            float cublas_alpha = 1.0;
            float cublas_beta = 0;
            timer.Start();
            for (int i = 0; i < repeat; i++) {
                cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 
                            &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
            }
            timer.Stop();
            time_cost = timer.Elapsed();
            cublasDestroy(cublas_handle); // Don't forget to destroy the handle
            break;
        }
        case EXECtileGemm: {
            const int tile_M_sz = 64;
            //const int tile_K_sz = 64;
            const int tile_N_sz = 64;
            int tile_M_num = (M + tile_M_sz - 1) / tile_M_sz;
            //int tile_K_num = (K + tile_K_sz - 1) / tile_K_sz;
            int tile_N_num = (N + tile_N_sz - 1) / tile_N_sz;
            const int blocksz = 256;
            dim3 gridsz(tile_M_num, tile_N_num);
            timer.Start();
            for (int i = 0; i < repeat; i++) {
                tile_sgemm_v2<<<gridsz, blocksz>>>(d_a, d_b, d_c, M, K, N);
            }
            cudaDeviceSynchronize();
            timer.Stop();
            time_cost = timer.Elapsed();
            break;
        }
        default:
            std::cout << "Unknown Execute Type" << std::endl;
            break;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    double avg_sec;
    avg_sec = time_cost / 1000.0 / repeat;
    avg_runtime = avg_sec;
    double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;
    return avg_Gflops;
}



void check_gemm(const float *h_a, const float *h_b, const float *h_check,
                const int size_a, const int size_b, const int size_c,
                const int M, const int K, const int N)
{

    
    //v2
    
    const int tile_M_sz = 64;
    const int tile_K_sz = 64;
    const int tile_N_sz = 64;
    
    

    //v1
    /*
    const int tile_M_sz = 64;
    const int tile_K_sz = WAPRSIZE;
    const int tile_N_sz = 64;
    */
    

    int tile_M_num = (M + tile_M_sz - 1) / tile_M_sz;
    int tile_K_num = (K + tile_K_sz - 1) / tile_K_sz;
    int tile_N_num = (N + tile_N_sz - 1) / tile_N_sz;

    
    printf("tile_M_num: %d ", tile_M_num);
    printf("tile_K_num: %d ", tile_K_num);
    printf("tile_N_num: %d\n", tile_N_num);
    

    GpuTimer timer;
    float time_cost;

    //const int blocksz = 256;
    const int blocksz = 256;
    dim3 gridsz(tile_M_num, tile_N_num);

    float *h_c;
    h_c = (float *)malloc(size_c);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemset(d_c, 0.0, size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    timer.Start();
    //tile_sgemm_v1<<<gridsz, blocksz>>>(d_a_tile, d_b_tile, d_c_tile, tile_M_num, tile_K_num, tile_N_num, M, K, N);
    // SGEMM::tile_sgemm_v1<<<gridsz, blocksz>>>(d_a_tile, d_b_tile, d_c_tile, tile_M_num, tile_K_num, tile_N_num, M, K, N);
    //tile_sgemm_v1<<<gridsz, blocksz>>>(d_a, d_b, d_c, M, K, N);
    //tile_sgemm_v2<<<gridsz, blocksz>>>(d_a, d_b, d_c, M, K, N);
    tile_sgemm_v3<<<gridsz, blocksz>>>(d_a, d_b, d_c, M, K, N);
    cudaDeviceSynchronize();
    timer.Stop();
    time_cost = timer.Elapsed();
    printf("gemm run on GPU: %f msecs.\n", time_cost);

    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    int pos = 0;
    for (int i = 0; i < M * N; i++)
    {
        float this_error = abs(h_check[i] - h_c[i]);
        if (max_error < this_error)
        {
            max_error = this_error;
            pos = i;
        }
        if (this_error >= 0.1)
            break;
        // std::cout<<i<<"-"<<h_c[i]<<"-"<<h_check[i]<<"  ";
    }
    printf("Max Error = %f\n", max_error);
    if (max_error > 0.1)
    {
        std::cout << "pos(i,j): " << pos/N <<" "<< pos%N << std::endl;
        for (int i = pos; i < pos + 5; i++)
        {
            if (i < M * N)
                std::cout << std::fixed << std::setprecision(6) << h_c[i] << "-" << h_check[i] << "  ";
        }
        std::cout << std::endl;
    }

    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void CublasSgemm(const float *h_a, const float *h_b, const float *h_check,
                 const int size_a, const int size_b, const int size_c,
                 const int M, const int K, const int N)
{

    GpuTimer timer;
    float time_cost;

    float *h_c, *d_a, *d_b, *d_c;

    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
    timer.Start();
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    timer.Stop();
    time_cost = timer.Elapsed();
    cublasDestroy(cublas_handle); // Don't forget to destroy the handle
    printf("cuBLAS gemm run on GPU: %f msecs.\n", time_cost);
    cudaMemcpy(h_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++)
    {
        float this_error = abs(h_check[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }
    printf("Max Error = %f\n", max_error);

    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

void check_acc(const float *h_a, const float *h_b,
               const int size_a, const int size_b, const int size_c,
               const int M, const int K, const int N)
{
    float *h_check;
    h_check = (float *)malloc(size_c);
    memset(h_check, 0, size_c);
    GpuTimer timer;
    float time_cost;
    timer.Start();
    cpuSgemm(h_a, h_b, h_check, M, N, K);
    timer.Stop();
    time_cost = timer.Elapsed();
    printf("gemm run on CPU: %f msecs.\n", time_cost);

    CublasSgemm(h_a, h_b, h_check, size_a, size_b, size_c, M, K, N);

    check_gemm(h_a, h_b, h_check, size_a, size_b, size_c, M, K, N);
    free(h_check);
}

void get_Performance(const float *h_a, const float *h_b,
               const int size_a, const int size_b, const int size_c,
               const int M, const int K, const int N)
{
    float avg_runtime;
    const int repeat = 20;
    double cublas_Performance = calc_Performance(h_a, h_b, size_a, size_b, size_c, M, N, K, repeat, EXECcuBLAS, avg_runtime);
    printf("AVG_cublas_RunTime = %lf sec\n", avg_runtime);
    double mygemm_Performance = calc_Performance(h_a, h_b, size_a, size_b, size_c, M, N, K, repeat, EXECtileGemm, avg_runtime);
    printf("AVG_mygemm_RunTime = %lf sec\n", avg_runtime);
    printf("AVG_cublas_Performance = %lf Gflops\n", cublas_Performance);
    printf("AVG_mygemm_Performance = %lf Gflops\n", mygemm_Performance);
}


void test_diff_size_gemm()
{
    int size_arr[10] = {1024, 1234, 2048, 2233, 4096, 5566, 8192, 8999, 10080, 12288};
    for (int i = 0; i < 10; i++)
    {
        const int M = size_arr[i];
        const int K = size_arr[i];
        const int N = size_arr[i];
        std::cout<<"Matrix Size : "<<M<<std::endl;

        float *h_a, *h_b;

        int size_a = M * K * sizeof(float);
        int size_b = K * N * sizeof(float);
        int size_c = M * N * sizeof(float);

        h_a = (float *)malloc(size_a);
        h_b = (float *)malloc(size_b);

        // h_a[i] = rand() / float(RAND_MAX);
        for (int i = 0; i < M * K; i++)
        {
            h_a[i] = rand() / float(RAND_MAX);
        }
            
        for (int i = 0; i < K * N; i++)
        {
            h_b[i] = rand() / float(RAND_MAX);
        }
            
        //check_acc(h_a, h_b, size_a, size_b, size_c, M, K, N);
        get_Performance(h_a, h_b, size_a, size_b, size_c, M, K, N);
        std::cout<<std::endl;
        
        free(h_a);
        free(h_b);
    }
    
}


void test_diff_size_gemm_acc()
{
    int size_arr[5] = {1024, 1234, 2048, 2345, 4096};
    for (int i = 0; i < 5; i++)
    {
        const int M = size_arr[i];
        const int K = size_arr[i];
        const int N = size_arr[i];
        std::cout<<"Matrix Size : "<<M<<std::endl;

        float *h_a, *h_b;

        int size_a = M * K * sizeof(float);
        int size_b = K * N * sizeof(float);
        int size_c = M * N * sizeof(float);

        h_a = (float *)malloc(size_a);
        h_b = (float *)malloc(size_b);

        // h_a[i] = rand() / float(RAND_MAX);
        for (int i = 0; i < M * K; i++)
        {
            h_a[i] = rand() / float(RAND_MAX);
        }
            
        for (int i = 0; i < K * N; i++)
        {
            h_b[i] = rand() / float(RAND_MAX);
        }
            
        check_acc(h_a, h_b, size_a, size_b, size_c, M, K, N);
        std::cout<<std::endl;
        free(h_a);
        free(h_b);
    }
    
}

