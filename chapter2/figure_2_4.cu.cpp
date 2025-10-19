#include <stddef.h>
#include <iostream>
#include <cuda_runtime.h>

void vecAdd(float* A_h, float* B_h, float* C_h, size_t n) {

  const auto size = n * sizeof(float);
  std::cout << "Allocating " << size << " bytes" << std::endl;

  float* A_d;
  float* B_d;
  float* C_d;
  cudaMalloc(&A_d, size);
  cudaMalloc(&B_d, size);
  cudaMalloc(&C_d, size);

  for (size_t i = 0; i < n; ++i) {
    C_h[i] = A_h[i] + B_h[i];
  }

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
