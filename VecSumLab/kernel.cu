#ifndef __CUDACC__  
#define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <random>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>

#define BLOCK_DIM_X 1000

using namespace std;

float get_sum_cpu(float* arr, int len) {
    float sum = 0;
    for (size_t i = 0; i < len; ++i)
        sum += arr[i];
    return sum;
}

__global__ void get_sum_gpu(float* arr, int len, float* res)
{
    __shared__ float temp[BLOCK_DIM_X];

    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < len) {
        temp[threadIdx.x] = arr[idx];
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        float summ = 0;
        for (int i = 0; i < blockDim.x; ++i)
            summ += temp[i];
        atomicAdd(res, summ);
    }

}

int main()
{
    srand(time(0));

    int len = 1000000;

    float* vec = new float[len];
    for (size_t i = 0; i < len; ++i)
        vec[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    clock_t start, end;
    float sum_cpu;

    start = clock();
    for (int i = 0; i < 12; ++i)
        sum_cpu = get_sum_cpu(vec, len);
    end = clock();

    double cpu_time = static_cast <double>(end - start) / static_cast <double>(CLOCKS_PER_SEC);

    std::cout << "\nSum on CPU:\t" << sum_cpu << "\nCPU time:\t" << cpu_time / 12;

    float* dvec;
    float* sum_gpu = new float;
    *sum_gpu = 0;
    float* dsum;
    cudaMalloc(&dvec, len * sizeof(float));
    cudaMalloc(&dsum, sizeof(float));

    cudaMemcpy(dvec, vec, len * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dsum, sum_gpu, sizeof(float), cudaMemcpyHostToDevice);

    dim3 block_dim(BLOCK_DIM_X);
    dim3 grid_dim(ceil(static_cast <float> (len) / static_cast <float> (block_dim.x)));

    cudaEvent_t begin, stop;
    cudaEventCreate(&begin);
    cudaEventCreate(&stop);

    cudaEventRecord(begin, 0);
    get_sum_gpu << <grid_dim, block_dim >> > (dvec, len, dsum);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpu_time;
    cudaEventElapsedTime(&gpu_time, begin, stop);

    cudaMemcpy(sum_gpu, dsum, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "\nSum on GPU:\t" << *sum_gpu << "\nGPU time:\t" << gpu_time / 1000.;

    cudaFree(dvec);

    
    thrust::device_vector<float> D(len);
    for (size_t i = 0; i < len; ++i)
        D[i] = vec[i];

    cudaEventRecord(begin, 0);
    float thrust_sum = thrust::reduce(D.begin(), D.end(), (float)0, thrust::plus<float>());
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float thrust_time;
    cudaEventElapsedTime(&thrust_time, begin, stop);
    std::cout << "\nSum on thrust:\t" << thrust_sum << "\nThrust time:\t" << thrust_time / 1000.;

    delete[] vec;

    return 0;
}