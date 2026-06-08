#include <stdio.h>
#include <stdlib.h>
// global tells run GPU
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main(){
    int N = 1000000;
    size_t size = N * sizeof(float);

    // allocate on CPU
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);
    // fill values
    for (int i = 0; i < N; i ++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // allocate GPU
    float* d_A;
    float* d_B;
    float* d_C;
    // cudaMalloc allocate mem on GPU instead of CPU
    cudaMalloc(&d_A, size); // pass address of pointer
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    //copy data from CPU memory to GPU memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // launch kernel
    int threadsPerBlock = 256; // 256 threatds per block
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // wait for GPU to finish
    cudaDeviceSynchronize();

     // copy result GPU to CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // verify
    printf("C[0] = %f\n", h_C[0]);
    printf("C[999999] = %f\n", h_C[999999]);

    // free everything
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
