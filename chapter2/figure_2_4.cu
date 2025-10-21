#include <stddef.h>
#include <iostream>

__global__
void vecAddKernel(float* A, float* B, float* C, size_t n) {
  const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  // printf("i %lu\n", i);
  if (i < n) {
    printf("i %lu\n", i);
    C[i] = A[i] + B[i];
  }
}

void vecAdd(float* A_h, float* B_h, float* C_h, size_t n) {

  const auto size = n * sizeof(float);
  std::cout << "Allocating " << size << " bytes" << std::endl;

  float* A_d;
  float* B_d;
  float* C_d;
  cudaMalloc(&A_d, size);
  cudaMalloc(&B_d, size);
  cudaMalloc(&C_d, size);

  const cudaError_t err_A = cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
  const cudaError_t err_B = cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);
  std::cout << cudaGetErrorString(err_A) << std::endl;
  std::cout << cudaGetErrorString(err_B) << std::endl;

  if (false) {
    std::cout << "Adding on CPU" << std::endl;
    for (size_t i = 0; i < n; ++i) {
      C_h[i] = A_h[i] + B_h[i];
    }
  } else {
    std::cout << "Adding on GPU" << std::endl;
    size_t n_blocks{4};
    size_t n_threads{256};
    vecAddKernel<<<n_blocks, n_threads>>>(A_d, B_d, C_d, n);
  }

  const cudaError_t err_C = cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
  std::cout << cudaGetErrorString(err_C) << std::endl;

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
  size_t n = 5;
  float A[n];
  float B[n];
  float C[n];
  A[0] = 1.0;
  A[1] = 3.0;
  A[2] = 5.0;
  A[3] = 7.0;
  A[4] = 9.0;
  B[0] = 10.0;
  B[1] = 30.0;
  B[2] = 50.0;
  B[3] = 70.0;
  B[4] = 90.0;
  vecAdd(A, B, C, n);
  for (size_t i = 0; i < n; ++i) {
    std::cout << i << " " << C[i] << std::endl;
  }
  return 0;
}
