#include <DNGemm.cuh>



__global__ void tile_sgemm_v0(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 64;
    const int TK = WAPRSIZE;
    const int TN = 64;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    const int tileB_sz = TN * TK;
    const int tileC_sz = TM * TN;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TK][TN];
    __shared__ float sh_tile_C[TM][TN];

    float fragA[4][4];
    float fragB[4][4];
    float fragC[4][4];

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
        sh_tile_B[i / TN][i % TN] = 0.0f;
    }
    for (int i = tid; i < tileC_sz; i += bsize)
    {
        sh_tile_C[i / TN][i % TN] = 0.0f;
    }
    __syncthreads();

    for (int i = tile_K - 1; i >= 0; i--)
    {
        // load matrix A to shared memory tile
        for (int j = tid; j < tileA_sz; j += bsize)
        {
            int i_row = j / TK;
            int i_col = j % TK;
            int g_row = global_tileA_rid + i_row;
            int g_col = i * TK + i_col;
            if (g_row < M && g_col < K)
                sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        }
        // load matrix B to shared memory tile
        for (int j = tid; j < tileB_sz; j += bsize)
        {
            int i_row = j / TN;
            int i_col = j % TN;
            int g_row = i * TK + i_row;
            int g_col = global_tileB_cid + i_col;
            if (g_row < K && g_col < N)
                sh_tile_B[i_row][i_col] = mtxB[g_row * N + g_col];
        }

#pragma unroll
        for (int k = 0; k < 4; k++)
        {
            fragC[k][0] = 0.0f;
            fragC[k][1] = 0.0f;
            fragC[k][2] = 0.0f;
            fragC[k][3] = 0.0f;
        }

        __syncthreads();

        // calc fragment matrix mut in register, the frag A size is 4x4, frag B size is 4x4
        // smem A is 64x32, B is 32x64, and we split the matrix A to 16x8 piece,
        // split matrix B to 8x16.
        // we use 256 threads calc it, each thread calc a piece row mut a piece col

        // tid/16
        int fragAid = tid / 16;
        // tid%16
        int fragBid = tid % 16;

        for (int j = 0; j < 8; j++)
        {
            // load frag A to register
            for (int k = 0; k < 4; k++)
            {
                fragA[k][0] = sh_tile_A[fragAid * 4 + k][j * 4];
                fragA[k][1] = sh_tile_A[fragAid * 4 + k][j * 4 + 1];
                fragA[k][2] = sh_tile_A[fragAid * 4 + k][j * 4 + 2];
                fragA[k][3] = sh_tile_A[fragAid * 4 + k][j * 4 + 3];
            }
            // load frag B to register
            for (int k = 0; k < 4; k++)
            {
                fragB[k][0] = sh_tile_B[j * 4 + k][fragBid * 4];
                fragB[k][1] = sh_tile_B[j * 4 + k][fragBid * 4 + 1];
                fragB[k][2] = sh_tile_B[j * 4 + k][fragBid * 4 + 2];
                fragB[k][3] = sh_tile_B[j * 4 + k][fragBid * 4 + 3];
            }

            // calc
            for (int k = 0; k < 4; k++)
            {
                for (int l = 0; l < 4; l++)
                {
                    float fmasum = 0.0;
                    fmasum += fragA[k][0] * fragB[0][l];
                    fmasum += fragA[k][1] * fragB[1][l];
                    fmasum += fragA[k][2] * fragB[2][l];
                    fmasum += fragA[k][3] * fragB[3][l];
                    fragC[k][l] += fmasum;
                }
            }
        }

        // write register results to shared mem
        for (int k = 0; k < 4; k++)
        {
            sh_tile_C[fragAid * 4 + k][fragBid * 4] += fragC[k][0];
            sh_tile_C[fragAid * 4 + k][fragBid * 4 + 1] += fragC[k][1];
            sh_tile_C[fragAid * 4 + k][fragBid * 4 + 2] += fragC[k][2];
            sh_tile_C[fragAid * 4 + k][fragBid * 4 + 3] += fragC[k][3];
        }

        __syncthreads();
    }

    __syncthreads();

    // write smem results back to global memory
    for (int i = tid; i < tileC_sz; i += bsize)
    {
        int g_rowId = i / TN;
        int g_colId = i % TN;
        int row_off = tile_row_id * TM + g_rowId;
        int col_off = tile_col_id * TN + g_colId;
        if (row_off < M && col_off < N)
            mtxC[row_off * N + col_off] = sh_tile_C[g_rowId][g_colId];
    }
}

__global__ void tile_sgemm_v1(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 64;
    const int TK = WAPRSIZE;
    const int TN = 64;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    const int tileB_sz = TN * TK;
    const int tileC_sz = TM * TN;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TK][TN];
    __shared__ float sh_tile_C[TM][TN];

    float4 fragA[4];
    float4 fragB[4];
    float4 fragC[4];

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
        sh_tile_B[i / TN][i % TN] = 0.0f;
    }
    for (int i = tid; i < tileC_sz; i += bsize)
    {
        sh_tile_C[i / TN][i % TN] = 0.0f;
    }
    __syncthreads();

    for (int i = tile_K - 1; i >= 0; i--)
    {
        // load matrix A to shared memory tile
        for (int j = tid; j < tileA_sz; j += bsize)
        {
            int i_row = j / TK;
            int i_col = j % TK;
            int g_row = global_tileA_rid + i_row;
            int g_col = i * TK + i_col;
            if (g_row < M && g_col < K)
                sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        }
        // load matrix B to shared memory tile
        for (int j = tid; j < tileB_sz; j += bsize)
        {
            int i_row = j / TN;
            int i_col = j % TN;
            int g_row = i * TK + i_row;
            int g_col = global_tileB_cid + i_col;
            if (g_row < K && g_col < N)
                sh_tile_B[i_row][i_col] = mtxB[g_row * N + g_col];
        }

#pragma unroll
        for (int k = 0; k < 4; k++)
        {
            fragC[k].x = 0.0f;
            fragC[k].y = 0.0f;
            fragC[k].z = 0.0f;
            fragC[k].w = 0.0f;
        }

        __syncthreads();

        // calc fragment matrix mut in register, the frag A size is 4x4, frag B size is 4x4
        // smem A is 64x32, B is 32x64, and we split the matrix A to 16x8 piece,
        // split matrix B to 8x16.
        // we use 256 threads calc it, each thread calc a piece row mut a piece col

        // tid/16
        int fragAid = tid / 16;
        // tid%16
        int fragBid = tid % 16;

        for (int j = 0; j < 8; j++)
        {
            // load frag A to register
            for (int k = 0; k < 4; k++)
            {
                fragA[k].x = sh_tile_A[fragAid * 4 + k][j * 4];
                fragA[k].y = sh_tile_A[fragAid * 4 + k][j * 4 + 1];
                fragA[k].z = sh_tile_A[fragAid * 4 + k][j * 4 + 2];
                fragA[k].w = sh_tile_A[fragAid * 4 + k][j * 4 + 3];
            }
            // load frag B to register
            for (int k = 0; k < 4; k++)
            {
                fragB[k].x = sh_tile_B[j * 4 + k][fragBid * 4];
                fragB[k].y = sh_tile_B[j * 4 + k][fragBid * 4 + 1];
                fragB[k].z = sh_tile_B[j * 4 + k][fragBid * 4 + 2];
                fragB[k].w = sh_tile_B[j * 4 + k][fragBid * 4 + 3];
            }

            // calc
            for (int k = 0; k < 4; k++)
            {
                float fmasum = 0.0;
                fmasum += fragA[k].x * fragB[0].x;
                fmasum += fragA[k].y * fragB[1].x;
                fmasum += fragA[k].z * fragB[2].x;
                fmasum += fragA[k].w * fragB[3].x;
                fragC[k].x += fmasum;

                fmasum = 0.0;
                fmasum += fragA[k].x * fragB[0].y;
                fmasum += fragA[k].y * fragB[1].y;
                fmasum += fragA[k].z * fragB[2].y;
                fmasum += fragA[k].w * fragB[3].y;
                fragC[k].y += fmasum;

                fmasum = 0.0;
                fmasum += fragA[k].x * fragB[0].z;
                fmasum += fragA[k].y * fragB[1].z;
                fmasum += fragA[k].z * fragB[2].z;
                fmasum += fragA[k].w * fragB[3].z;
                fragC[k].z += fmasum;

                fmasum = 0.0;
                fmasum += fragA[k].x * fragB[0].w;
                fmasum += fragA[k].y * fragB[1].w;
                fmasum += fragA[k].z * fragB[2].w;
                fmasum += fragA[k].w * fragB[3].w;
                fragC[k].w += fmasum;
            }
        }

        // write register results to shared mem
        for (int k = 0; k < 4; k++)
        {
            sh_tile_C[fragAid * 4 + k][fragBid * 4] += fragC[k].x;
            sh_tile_C[fragAid * 4 + k][fragBid * 4 + 1] += fragC[k].y;
            sh_tile_C[fragAid * 4 + k][fragBid * 4 + 2] += fragC[k].z;
            sh_tile_C[fragAid * 4 + k][fragBid * 4 + 3] += fragC[k].w;
        }

        __syncthreads();
    }

    __syncthreads();

    // write smem results back to global memory
    for (int i = tid; i < tileC_sz; i += bsize)
    {
        int g_rowId = i / TN;
        int g_colId = i % TN;
        int row_off = tile_row_id * TM + g_rowId;
        int col_off = tile_col_id * TN + g_colId;
        if (row_off < M && col_off < N)
            mtxC[row_off * N + col_off] = sh_tile_C[g_rowId][g_colId];
    }
}

__global__ void tile_sgemm_v2(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 64;
    const int TK = 64;
    const int TN = 64;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    const int tileB_sz = TN * TK;
    const int tileC_sz = TM * TN;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TK][TN];
    __shared__ float sh_tile_C[TM][TN];

    float fragA[4][16];
    float fragB[16][4];
    float fragC[4][4];

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
        sh_tile_B[i / TN][i % TN] = 0.0f;
    }
    for (int i = tid; i < tileC_sz; i += bsize)
    {
        sh_tile_C[i / TN][i % TN] = 0.0f;
    }
    __syncthreads();

    for (int i = tile_K - 1; i >= 0; i--)
    {

        // load matrix A to shared memory tile
        for (int j = tid; j < tileA_sz; j += bsize)
        {
            int i_row = j / TK;
            int i_col = j % TK;
            int g_row = global_tileA_rid + i_row;
            int g_col = i * TK + i_col;
            if (g_row < M && g_col < K)
                sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        }
        // load matrix B to shared memory tile
        for (int j = tid; j < tileB_sz; j += bsize)
        {
            int i_row = j / TN;
            int i_col = j % TN;
            int g_row = i * TK + i_row;
            int g_col = global_tileB_cid + i_col;
            if (g_row < K && g_col < N)
                sh_tile_B[i_row][i_col] = mtxB[g_row * N + g_col];
        }

#pragma unroll
        for (int k = 0; k < 4; k++)
        {
            for (int l = 0; l < 4; l++)
            {
                fragC[k][l] = 0.0f;
            }
        }

        __syncthreads();

        // calc fragment matrix mut in register, the frag A size is 4x4, frag B size is 4x4
        // smem A is 64x32, B is 32x64, and we split the matrix A to 16x8 piece,
        // split matrix B to 8x16.
        // we use 256 threads calc it, each thread calc a piece row mut a piece col

        // tid/16
        int fragAid = tid / 16;
        // tid%16
        int fragBid = tid % 16;

        for (int j = 0; j < 4; j++)
        {
            // load frag A to register
            for (int k = 0; k < 4; k++)
            {
                for (int l = 0; l < 16; l++)
                {
                    fragA[k][l] = sh_tile_A[fragAid * 4 + k][j * 16 + l];
                }
            }
            // load frag B to register
            for (int k = 0; k < 16; k++)
            {
                for (int l = 0; l < 4; l++)
                {
                    fragB[k][l] = sh_tile_B[j * 16 + k][fragBid * 4 + l];
                }
            }

            // calc
            for (int k = 0; k < 4; k++)
            {
                for (int l = 0; l < 4; l++)
                {
                    float fmasum = 0.0;
                    for (int u = 0; u < 16; u++)
                    {
                        fmasum += fragA[k][u] * fragB[u][l];
                    }
                    fragC[k][l] += fmasum;
                }
            }
        }

        // write register results to shared mem
        for (int k = 0; k < 4; k++)
        {
            for (int l = 0; l < 4; l++)
            {
                sh_tile_C[fragAid * 4 + k][fragBid * 4 + l] += fragC[k][l];
            }
        }

        __syncthreads();
    }

    __syncthreads();

    // write smem results back to global memory
    for (int i = tid; i < tileC_sz; i += bsize)
    {
        int g_rowId = i / TN;
        int g_colId = i % TN;
        int row_off = tile_row_id * TM + g_rowId;
        int col_off = tile_col_id * TN + g_colId;
        if (row_off < M && col_off < N)
            mtxC[row_off * N + col_off] = sh_tile_C[g_rowId][g_colId];
    }
}

__global__ void tile_sgemm_v3(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 64;
    const int TK = 64;
    const int TN = 64;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    const int tileB_sz = TN * TK;
    //const int tileC_sz = TM * TN;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TK][TN];

    float4 fragA = {0.0f,0.0f,0.0f,0.0f};
    float4 fragB[4] = {0.0f};
    float fragC[4][4] = {0.0f};

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
        sh_tile_B[i / TN][i % TN] = 0.0f;
    }
    __syncthreads();

    // tid/16
    int fragAid = tid / 16;
    // tid%16
    int fragBid = tid % 16;

    for (int i = tile_K - 1; i >= 0; i--)
    {

        // load matrix A to shared memory tile
        for (int j = tid; j < tileA_sz; j += bsize)
        {
            int i_row = j / TK;
            int i_col = j % TK;
            int g_row = global_tileA_rid + i_row;
            int g_col = i * TK + i_col;
            if (g_row < M && g_col < K)
                sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        }
        // load matrix B to shared memory tile
        for (int j = tid; j < tileB_sz; j += bsize)
        {
            int i_row = j / TN;
            int i_col = j % TN;
            int g_row = i * TK + i_row;
            int g_col = global_tileB_cid + i_col;
            if (g_row < K && g_col < N)
                sh_tile_B[i_row][i_col] = mtxB[g_row * N + g_col];
        }

        __syncthreads();

        // calc fragment matrix mut in register, the frag A size is 4x4, frag B size is 4x4
        // smem A is 64x32, B is 32x64, and we split the matrix A to 16x8 piece,
        // split matrix B to 8x16.
        // we use 256 threads calc it, each thread calc a piece row mut a piece col

        for (int j = 0; j < 16; j++)
        {   
            // load frag B to register
            for (int k = 0; k < 4; k++)
            {
                fragB[k] = FLOAT4(sh_tile_B[j * 4 + k][fragBid*4]);
            }
            //load frag A and calc
            for (int k = 0; k < 4; k++)
            {
                fragA = FLOAT4(sh_tile_A[fragAid * 4 + k][j*4]);

                fragC[k][0] += fragA.x * fragB[0].x;
                fragC[k][0] += fragA.y * fragB[1].x;
                fragC[k][0] += fragA.z * fragB[2].x;
                fragC[k][0] += fragA.w * fragB[3].x;

                fragC[k][1] += fragA.x * fragB[0].y;
                fragC[k][1] += fragA.y * fragB[1].y;
                fragC[k][1] += fragA.z * fragB[2].y;
                fragC[k][1] += fragA.w * fragB[3].y;

                fragC[k][2] += fragA.x * fragB[0].z;
                fragC[k][2] += fragA.y * fragB[1].z;
                fragC[k][2] += fragA.z * fragB[2].z;
                fragC[k][2] += fragA.w * fragB[3].z;

                fragC[k][3] += fragA.x * fragB[0].w;
                fragC[k][3] += fragA.y * fragB[1].w;
                fragC[k][3] += fragA.z * fragB[2].w;
                fragC[k][3] += fragA.w * fragB[3].w;
            }
            
        }

        __syncthreads();
    }

    __syncthreads();

    // write reg back to global 
    int row_off = tile_row_id * TM;
    int col_off = tile_col_id * TN;
    for (int k = 0; k < 4; k++)
    {
        for (int l = 0; l < 4; l++)
        {
            int w_grow_id = row_off+fragAid * 4 + k;
            int w_gcol_id = col_off+fragBid * 4 + l;
            if(w_grow_id < M && w_gcol_id < N)
                mtxC[w_grow_id * N + w_gcol_id] = fragC[k][l];
        }
    }
}


__global__ void tile_sgemm_v4(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 64;
    const int TK = 16;
    const int TN = 64;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    const int tileB_sz = TN * TK;
    //const int tileC_sz = TM * TN;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    const int frag_itNum = TK/4;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TK][TN];

    float4 fragA = {0.0f,0.0f,0.0f,0.0f};
    float4 fragB[4] = {0.0f};
    float fragC[4][4] = {0.0f};

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
        sh_tile_B[i / TN][i % TN] = 0.0f;
    }
    __syncthreads();

    // tid/16
    int fragAid = tid / 16;
    // tid%16
    int fragBid = tid % 16;

    for (int i = tile_K - 1; i >= 0; i--)
    {

        // load matrix A to shared memory tile
        for (int j = tid; j < tileA_sz; j += bsize)
        {
            int i_row = j / TK;
            int i_col = j % TK;
            int g_row = global_tileA_rid + i_row;
            int g_col = i * TK + i_col;
            if (g_row < M && g_col < K)
                sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        }
        // load matrix B to shared memory tile
        for (int j = tid; j < tileB_sz; j += bsize)
        {
            int i_row = j / TN;
            int i_col = j % TN;
            int g_row = i * TK + i_row;
            int g_col = global_tileB_cid + i_col;
            if (g_row < K && g_col < N)
                sh_tile_B[i_row][i_col] = mtxB[g_row * N + g_col];
        }

        __syncthreads();

        // calc fragment matrix mut in register, the frag A size is 4x4, frag B size is 4x4
        // smem A is 64x32, B is 32x64, and we split the matrix A to 16x8 piece,
        // split matrix B to 8x16.
        // we use 256 threads calc it, each thread calc a piece row mut a piece col

        for (int j = 0; j < frag_itNum; j++)
        {   
            // load frag B to register
            for (int k = 0; k < 4; k++)
            {
                fragB[k] = FLOAT4(sh_tile_B[j * 4 + k][fragBid*4]);
            }
            //load frag A and calc
            for (int k = 0; k < 4; k++)
            {
                fragA = FLOAT4(sh_tile_A[fragAid * 4 + k][j*4]);

                fragC[k][0] += fragA.x * fragB[0].x;
                fragC[k][0] += fragA.y * fragB[1].x;
                fragC[k][0] += fragA.z * fragB[2].x;
                fragC[k][0] += fragA.w * fragB[3].x;

                fragC[k][1] += fragA.x * fragB[0].y;
                fragC[k][1] += fragA.y * fragB[1].y;
                fragC[k][1] += fragA.z * fragB[2].y;
                fragC[k][1] += fragA.w * fragB[3].y;

                fragC[k][2] += fragA.x * fragB[0].z;
                fragC[k][2] += fragA.y * fragB[1].z;
                fragC[k][2] += fragA.z * fragB[2].z;
                fragC[k][2] += fragA.w * fragB[3].z;

                fragC[k][3] += fragA.x * fragB[0].w;
                fragC[k][3] += fragA.y * fragB[1].w;
                fragC[k][3] += fragA.z * fragB[2].w;
                fragC[k][3] += fragA.w * fragB[3].w;
            }
            
        }

        __syncthreads();
    }

    __syncthreads();

    // write reg back to global 
    int row_off = tile_row_id * TM;
    int col_off = tile_col_id * TN;
    for (int k = 0; k < 4; k++)
    {
        for (int l = 0; l < 4; l++)
        {
            int w_grow_id = row_off+fragAid * 4 + k;
            int w_gcol_id = col_off+fragBid * 4 + l;
            if(w_grow_id < M && w_gcol_id < N)
                mtxC[w_grow_id * N + w_gcol_id] = fragC[k][l];
        }
    }
}



__global__ void tile_sgemm_vf(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 128;
    const int TK = 8;
    const int TN = 128;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    //const int tileB_sz = TN * TK;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TK][TN];

   
    const int RM = 8;
    //const int RK = 8;
    const int RN = 8;
    float4 fragA = {0.0f};
    float4 fragB[8] = {0.0f};
    float fragC[RM][RN] = {0.0f};

    // tid/16
    int fragAid = tid / 16;
    // tid%16
    int fragBid = tid % 16;

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
        sh_tile_B[i / TN][i % TN] = 0.0f;
    }
    __syncthreads();

    for (int i = tile_K - 1; i >= 0; i--)
    {

        // load matrix A to shared memory tile
        int i_row = tid / 2;
        int i_col = (tid % 2)*4;
        int g_row = global_tileA_rid + i_row;
        int g_col = i * TK + i_col;

        FLOAT4(sh_tile_A[i_row][i_col]) = FLOAT4(mtxA[g_row * K + g_col]);

        /*
        if (g_row < M && g_col < K)
            sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        if (g_row < M && g_col+1 < K)
            sh_tile_A[i_row][i_col+1] = mtxA[g_row * K + g_col+1];
        if (g_row < M && g_col+2 < K)
            sh_tile_A[i_row][i_col+2] = mtxA[g_row * K + g_col+2];
        if (g_row < M && g_col+3 < K)
            sh_tile_A[i_row][i_col+3] = mtxA[g_row * K + g_col+3];
        */

        // load matrix B to shared memory tile
        i_row = tid / 32;
        i_col = (tid % 32)*4;
        g_row = i * TK + i_row;
        g_col = global_tileB_cid + i_col;

        FLOAT4(sh_tile_B[i_row][i_col]) = FLOAT4(mtxB[g_row * N + g_col]);

        /*
        if (g_row < K && g_col < N)
            sh_tile_B[i_row][i_col] = mtxB[g_row * N + g_col];
        if (g_row < K && g_col+1 < N)
            sh_tile_B[i_row][i_col+1] = mtxB[g_row * N + g_col+1];
        if (g_row < K && g_col+2 < N)
            sh_tile_B[i_row][i_col+2] = mtxB[g_row * N + g_col+2];
        if (g_row < K && g_col+3 < N)
            sh_tile_B[i_row][i_col+3] = mtxB[g_row * N + g_col+3];
        */

        __syncthreads();


        // we use 256 threads calc it, each thread calc a 8x8 mat mut
        // load frag B to register
        for (int l = 0; l < 2; l++)
        {
            for (int k = 0; k < 8; k++)
            {
                fragB[k] = FLOAT4(sh_tile_B[k][fragBid*8+l*4]);
            }
            //load frag A and calc
            for (int k = 0; k < 8; k++)
            {
                fragA = FLOAT4(sh_tile_A[fragAid*8+k][0]);

                fragC[k][l*4] += fragA.x*fragB[0].x;
                fragC[k][l*4] += fragA.y*fragB[1].x;
                fragC[k][l*4] += fragA.z*fragB[2].x;
                fragC[k][l*4] += fragA.w*fragB[3].x;

                fragC[k][l*4+1] += fragA.x*fragB[0].y;
                fragC[k][l*4+1] += fragA.y*fragB[1].y;
                fragC[k][l*4+1] += fragA.z*fragB[2].y;
                fragC[k][l*4+1] += fragA.w*fragB[3].y;

                fragC[k][l*4+2] += fragA.x*fragB[0].z;
                fragC[k][l*4+2] += fragA.y*fragB[1].z;
                fragC[k][l*4+2] += fragA.z*fragB[2].z;
                fragC[k][l*4+2] += fragA.w*fragB[3].z;

                fragC[k][l*4+3] += fragA.x*fragB[0].w;
                fragC[k][l*4+3] += fragA.y*fragB[1].w;
                fragC[k][l*4+3] += fragA.z*fragB[2].w;
                fragC[k][l*4+3] += fragA.w*fragB[3].w;


                fragA = FLOAT4(sh_tile_A[fragAid*8+k][4]);

                fragC[k][l*4] += fragA.x*fragB[4].x;
                fragC[k][l*4] += fragA.y*fragB[5].x;
                fragC[k][l*4] += fragA.z*fragB[6].x;
                fragC[k][l*4] += fragA.w*fragB[7].x;

                fragC[k][l*4+1] += fragA.x*fragB[4].y;
                fragC[k][l*4+1] += fragA.y*fragB[5].y;
                fragC[k][l*4+1] += fragA.z*fragB[6].y;
                fragC[k][l*4+1] += fragA.w*fragB[7].y;

                fragC[k][l*4+2] += fragA.x*fragB[4].z;
                fragC[k][l*4+2] += fragA.y*fragB[5].z;
                fragC[k][l*4+2] += fragA.z*fragB[6].z;
                fragC[k][l*4+2] += fragA.w*fragB[7].z;
                
                fragC[k][l*4+3] += fragA.x*fragB[4].w;
                fragC[k][l*4+3] += fragA.y*fragB[5].w;
                fragC[k][l*4+3] += fragA.z*fragB[6].w;
                fragC[k][l*4+3] += fragA.w*fragB[7].w;

            }
        }
        
        
        
        __syncthreads();
    }

    // write reg back to global 
    int row_off = tile_row_id * TM;
    int col_off = tile_col_id * TN;
    for (int k = 0; k < RM; k++)
    {
        for (int l = 0; l < RN; l++)
        {
            int w_grow_id = row_off+fragAid * RM + k;
            int w_gcol_id = col_off+fragBid * RN + l;
            if(w_grow_id < M && w_gcol_id < N)
                mtxC[w_grow_id * N + w_gcol_id] = fragC[k][l];
        }
    }
    
}



__global__ void tile_sgemm_vf2(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 128;
    const int TK = 8;
    const int TN = 128;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    //const int tileB_sz = TN * TK;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TK][TN];

   
    const int RM = 8;
    //const int RK = 8;
    const int RN = 8;

    float4 fragA[2] = {0.0f};
    //float4 fragB[2] = {0.0f};
    float fragC[RM][RN] = {0.0f};

    int warp_id = tid/32;
    int lane_id = tid%32;

    int outer_warp_xid = warp_id/2;
    int outer_warp_yid = warp_id%2;

    int inner_warp_xid = lane_id/8;
    int inner_warp_yid = lane_id%8;

    int sh_A_row_star = outer_warp_xid*32+inner_warp_xid*8;
    int sh_B_col_star = outer_warp_yid*64+inner_warp_yid*8;

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
        sh_tile_B[i / TN][i % TN] = 0.0f;
    }
    __syncthreads();

    for (int i = tile_K - 1; i >= 0; i--)
    {

        // load matrix A to shared memory tile
        int i_row = tid / 2;
        int i_col = (tid % 2)*4;
        int g_row = global_tileA_rid + i_row;
        int g_col = i * TK + i_col;
        if (g_row < M && g_col < K)
            sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        if (g_row < M && g_col+1 < K)
            sh_tile_A[i_row][i_col+1] = mtxA[g_row * K + g_col+1];
        if (g_row < M && g_col+2 < K)
            sh_tile_A[i_row][i_col+2] = mtxA[g_row * K + g_col+2];
        if (g_row < M && g_col+3 < K)
            sh_tile_A[i_row][i_col+3] = mtxA[g_row * K + g_col+3];

        // load matrix B to shared memory tile
        i_row = tid / 32;
        i_col = (tid % 32)*4;
        g_row = i * TK + i_row;
        g_col = global_tileB_cid + i_col;
        if (g_row < K && g_col < N)
            sh_tile_B[i_row][i_col] = mtxB[g_row * N + g_col];
        if (g_row < K && g_col+1 < N)
            sh_tile_B[i_row][i_col+1] = mtxB[g_row * N + g_col+1];
        if (g_row < K && g_col+2 < N)
            sh_tile_B[i_row][i_col+2] = mtxB[g_row * N + g_col+2];
        if (g_row < K && g_col+3 < N)
            sh_tile_B[i_row][i_col+3] = mtxB[g_row * N + g_col+3];

        __syncthreads();


        // use 256 threads calc it, each thread calc a 8x8 mat mut
        /*
            for (int j = 0; j < RM; j++)
            {
                for (int k = 0; k < RN; k++)
                {
                    for (int l = 0; l < RK; l++)
                    {
                    
                        fragC[j][k] += sh_tile_A[sh_A_row_star+j][l] * sh_tile_B[l][sh_B_col_star+k];
                    } 
                }
            
            }

        */

        for (int j = 0; j < RM; j++)
        {
            for (int k = 0; k < RN; k++)
            {
                fragA[0] = FLOAT4(sh_tile_A[sh_A_row_star+j][0]);
                fragA[1] = FLOAT4(sh_tile_A[sh_A_row_star+j][4]);

                fragC[j][k] += fragA[0].x * sh_tile_B[0][sh_B_col_star+k];
                fragC[j][k] += fragA[0].y * sh_tile_B[1][sh_B_col_star+k];
                fragC[j][k] += fragA[0].z * sh_tile_B[2][sh_B_col_star+k];
                fragC[j][k] += fragA[0].w * sh_tile_B[3][sh_B_col_star+k];

                fragC[j][k] += fragA[1].x * sh_tile_B[4][sh_B_col_star+k];
                fragC[j][k] += fragA[1].y * sh_tile_B[5][sh_B_col_star+k];
                fragC[j][k] += fragA[1].z * sh_tile_B[6][sh_B_col_star+k];
                fragC[j][k] += fragA[1].w * sh_tile_B[7][sh_B_col_star+k];
            }
            
        }
        
        __syncthreads();
    }

    // write reg back to global 
    int row_off = tile_row_id * TM;
    int col_off = tile_col_id * TN;
    int w_grow_id = row_off+outer_warp_xid*32+inner_warp_xid*8;
    int w_gcol_id = col_off+outer_warp_yid*64+inner_warp_yid*8;
    for (int k = 0; k < RM; k++)
    {
        for (int l = 0; l < RN; l++)
        {
            if(w_grow_id+k < M && w_gcol_id+l < N)
                mtxC[(w_grow_id+k) * N + w_gcol_id+l] = fragC[k][l];
        }
    }
    
}



__global__ void tile_sgemm_vf2_trans(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 128;
    const int TK = 8;
    const int TN = 128;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    //const int tileB_sz = TN * TK;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TN][TK];

   
    const int RM = 8;
    //const int RK = 8;
    const int RN = 8;

    float4 fragA;
    float4 fragB;
    float fragC[RM][RN] = {0.0f};

    int warp_id = tid/32;
    int lane_id = tid%32;

    int outer_warp_xid = warp_id/2;
    int outer_warp_yid = warp_id%2;

    int inner_warp_xid = lane_id/8;
    int inner_warp_yid = lane_id%8;

    int sh_A_row_star = outer_warp_xid*32+inner_warp_xid*8;
    int sh_B_col_star = outer_warp_yid*64+inner_warp_yid*8;

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
        sh_tile_B[i / TK][i % TK] = 0.0f;
    }
    __syncthreads();

    for (int i = tile_K - 1; i >= 0; i--)
    {

        // load matrix A to shared memory tile
        int i_row = tid / 2;
        int i_col = (tid % 2)*4;
        int g_row = global_tileA_rid + i_row;
        int g_col = i * TK + i_col;
        if (g_row < M && g_col < K)
            sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        if (g_row < M && g_col+1 < K)
            sh_tile_A[i_row][i_col+1] = mtxA[g_row * K + g_col+1];
        if (g_row < M && g_col+2 < K)
            sh_tile_A[i_row][i_col+2] = mtxA[g_row * K + g_col+2];
        if (g_row < M && g_col+3 < K)
            sh_tile_A[i_row][i_col+3] = mtxA[g_row * K + g_col+3];

        // load matrix B to shared memory tile
        i_row = tid / 32;
        i_col = (tid % 32)*4;
        g_row = i * TK + i_row;
        g_col = global_tileB_cid + i_col;
        if (g_row < K && g_col < N)
            sh_tile_B[i_col][i_row] = mtxB[g_row * N + g_col];
        if (g_row < K && g_col+1 < N)
            sh_tile_B[i_col+1][i_row] = mtxB[g_row * N + g_col+1];
        if (g_row < K && g_col+2 < N)
            sh_tile_B[i_col+2][i_row] = mtxB[g_row * N + g_col+2];
        if (g_row < K && g_col+3 < N)
            sh_tile_B[i_col+3][i_row] = mtxB[g_row * N + g_col+3];

        __syncthreads();


        // use 256 threads calc it, each thread calc a 8x8 mat mut
        /*
            for (int j = 0; j < RM; j++)
            {
                for (int k = 0; k < RN; k++)
                {
                    for (int l = 0; l < RK; l++)
                    {
                    
                        fragC[j][k] += sh_tile_A[sh_A_row_star+j][l] * sh_tile_B[l][sh_B_col_star+k];
                    } 
                }
            
            }

        */

        for (int j = 0; j < RM; j++)
        {
            for (int k = 0; k < RN; k++)
            {
                fragA = FLOAT4(sh_tile_A[sh_A_row_star+j][0]);
                fragB = FLOAT4(sh_tile_B[sh_B_col_star+k][0]);
                

                fragC[j][k] += fragA.x * fragB.x;
                fragC[j][k] += fragA.y * fragB.y;
                fragC[j][k] += fragA.z * fragB.z;
                fragC[j][k] += fragA.w * fragB.w;

                fragA = FLOAT4(sh_tile_A[sh_A_row_star+j][4]);
                fragB = FLOAT4(sh_tile_B[sh_B_col_star+k][4]);

                fragC[j][k] += fragA.x * fragB.x;
                fragC[j][k] += fragA.y * fragB.y;
                fragC[j][k] += fragA.z * fragB.z;
                fragC[j][k] += fragA.w * fragB.w;
            }
            
        }
        
        __syncthreads();
    }

    // write reg back to global 
    int row_off = tile_row_id * TM;
    int col_off = tile_col_id * TN;
    int w_grow_id = row_off+outer_warp_xid*32+inner_warp_xid*8;
    int w_gcol_id = col_off+outer_warp_yid*64+inner_warp_yid*8;
    for (int k = 0; k < RM; k++)
    {
        for (int l = 0; l < RN; l++)
        {
            if(w_grow_id+k < M && w_gcol_id+l < N)
                mtxC[(w_grow_id+k) * N + w_gcol_id+l] = fragC[k][l];
        }
    }
    
}


__global__ void tile_sgemm_vf3(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N)
{
    const int TM = 128;
    const int TK = 8;
    const int TN = 64;

    const int tid = threadIdx.x;
    // 256
    const int bsize = blockDim.x;
    const int tile_row_id = blockIdx.x;
    const int tile_col_id = blockIdx.y;
    const int tile_K = (K + TK - 1) / TK;

    const int tileA_sz = TM * TK;
    const int tileB_sz = TN * TK;
    const int global_tileA_rid = tile_row_id * TM;
    const int global_tileB_cid = tile_col_id * TN;

    __shared__ float sh_tile_A[TM][TK];
    __shared__ float sh_tile_B[TK][TN];

   
    const int RM = 8;
    //const int RK = 8;
    const int RN = 4;
    float4 fragA = {0.0f};
    float4 fragB[8] = {0.0f};
    float fragC[RM][RN] = {0.0f};

    // tid/16
    int fragAid = tid / 16;
    // tid%16
    int fragBid = tid % 16;

    for (int i = tid; i < tileA_sz; i += bsize)
    {
        sh_tile_A[i / TK][i % TK] = 0.0f;
    }
    for (int i = tid; i < tileB_sz; i += bsize)
    {
        sh_tile_B[i / TN][i % TN] = 0.0f;
    }
    __syncthreads();

    for (int i = tile_K - 1; i >= 0; i--)
    {

        // load matrix A to shared memory tile
        int i_row = tid / 2;
        int i_col = (tid % 2)*4;
        int g_row = global_tileA_rid + i_row;
        int g_col = i * TK + i_col;
        FLOAT4(sh_tile_A[i_row][i_col]) = FLOAT4(mtxA[g_row * K + g_col]);
        /*
        if (g_row < M && g_col < K)
            sh_tile_A[i_row][i_col] = mtxA[g_row * K + g_col];
        if (g_row < M && g_col+1 < K)
            sh_tile_A[i_row][i_col+1] = mtxA[g_row * K + g_col+1];
        if (g_row < M && g_col+2 < K)
            sh_tile_A[i_row][i_col+2] = mtxA[g_row * K + g_col+2];
        if (g_row < M && g_col+3 < K)
            sh_tile_A[i_row][i_col+3] = mtxA[g_row * K + g_col+3];
            */
        

        // load matrix B to shared memory tile
        i_row = tid / 32;
        i_col = (tid % 32)*2;
        g_row = i * TK + i_row;
        g_col = global_tileB_cid + i_col;
        FLOAT2(sh_tile_B[i_row][i_col]) = FLOAT2(mtxB[g_row * N + g_col]);
        /*
        if (g_row < K && g_col < N)
            sh_tile_B[i_row][i_col] = mtxB[g_row * N + g_col];
        if (g_row < K && g_col+1 < N)
            sh_tile_B[i_row][i_col+1] = mtxB[g_row * N + g_col+1];
            */
        
        __syncthreads();


        // we use 256 threads calc it, each thread calc a 8x8 mat mut
        // load frag B to register
        for (int k = 0; k < 8; k++)
        {
            fragB[k] = FLOAT4(sh_tile_B[k][fragBid*4]);
        }
        //load frag A and calc
        for (int k = 0; k < 8; k++)
        {
            fragA = FLOAT4(sh_tile_A[fragAid*8+k][0]);

            fragC[k][0] += fragA.x*fragB[0].x;
            fragC[k][0] += fragA.y*fragB[1].x;
            fragC[k][0] += fragA.z*fragB[2].x;
            fragC[k][0] += fragA.w*fragB[3].x;

            fragC[k][1] += fragA.x*fragB[0].y;
            fragC[k][1] += fragA.y*fragB[1].y;
            fragC[k][1] += fragA.z*fragB[2].y;
            fragC[k][1] += fragA.w*fragB[3].y;

            fragC[k][2] += fragA.x*fragB[0].z;
            fragC[k][2] += fragA.y*fragB[1].z;
            fragC[k][2] += fragA.z*fragB[2].z;
            fragC[k][2] += fragA.w*fragB[3].z;

            fragC[k][3] += fragA.x*fragB[0].w;
            fragC[k][3] += fragA.y*fragB[1].w;
            fragC[k][3] += fragA.z*fragB[2].w;
            fragC[k][3] += fragA.w*fragB[3].w;


            fragA = FLOAT4(sh_tile_A[fragAid*8+k][4]);

            fragC[k][0] += fragA.x*fragB[4].x;
            fragC[k][0] += fragA.y*fragB[5].x;
            fragC[k][0] += fragA.z*fragB[6].x;
            fragC[k][0] += fragA.w*fragB[7].x;
            
            fragC[k][1] += fragA.x*fragB[4].y;
            fragC[k][1] += fragA.y*fragB[5].y;
            fragC[k][1] += fragA.z*fragB[6].y;
            fragC[k][1] += fragA.w*fragB[7].y;

            fragC[k][2] += fragA.x*fragB[4].z;
            fragC[k][2] += fragA.y*fragB[5].z;
            fragC[k][2] += fragA.z*fragB[6].z;
            fragC[k][2] += fragA.w*fragB[7].z;
            
            fragC[k][3] += fragA.x*fragB[4].w;
            fragC[k][3] += fragA.y*fragB[5].w;
            fragC[k][3] += fragA.z*fragB[6].w;
            fragC[k][3] += fragA.w*fragB[7].w;

        }
        
        __syncthreads();
    }

    // write reg back to global 
    int row_off = tile_row_id * TM;
    int col_off = tile_col_id * TN;
    for (int k = 0; k < RM; k++)
    {
        for (int l = 0; l < RN; l++)
        {
            int w_grow_id = row_off+fragAid * RM + k;
            int w_gcol_id = col_off+fragBid * RN + l;
            if(w_grow_id < M && w_gcol_id < N)
                mtxC[w_grow_id * N + w_gcol_id] = fragC[k][l];
        }
    }
    
}
