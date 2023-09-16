#ifndef GPU_DNGEMM_H__
#define GPU_DNGEMM_H__

#include <common.cuh>
#include <timer.cuh>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>



__global__ void tile_sgemm_v0(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N);

__global__ void tile_sgemm_v1(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N);

__global__ void tile_sgemm_v2(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N);

__global__ void tile_sgemm_v3(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N);

__global__ void tile_sgemm_vf(float *__restrict__ mtxA, float *__restrict__ mtxB,
                              float *__restrict__ mtxC,
                              const int M, const int K, const int N);

#endif