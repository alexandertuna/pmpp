#include <vector>
#include <iostream>

std::vector<float> get_image();

__global__
void blur_image_kernel() {
  const unsigned int idx_x = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int idx_y = blockDim.y * blockIdx.y + threadIdx.y;
  // printf("blur_image_kernel %u %u\n", idx_x, idx_y);
  // get linearized coordinate
  // blur it
}

std::vector<float> blur_image(float* image_h,
                              const size_t nrows,
                              const size_t ncols) {
  const size_t ntotal{nrows * ncols};
  const size_t size{ntotal * sizeof(float)};

  // set up device
  float* image_d;
  cudaMalloc(&image_d, size);
  const cudaError_t err = cudaMemcpy(image_d, image_h, size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    const auto msg = cudaGetErrorString(err);
    throw std::runtime_error(msg);
  }

  // set up threads
  if (nrows == 0 or ncols == 0) {
    throw std::runtime_error("I cant work with this");
  }
  const size_t nthreads_x{8};
  const size_t nthreads_y{4};
  const size_t nthreads_z{1};
  const size_t nblocks_x{(nrows - 1) / nthreads_x + 1};
  const size_t nblocks_y{(ncols - 1) / nthreads_y + 1};
  const size_t nblocks_z{1};
  printf("nrows=%i ncols=%i -> nthreads_x=%i nthreads_y=%i nblocks_x=%i nblocks_x=%i\n",
         nrows, ncols, nthreads_x, nthreads_y, nblocks_x, nblocks_y);
  const dim3 nthreads(nthreads_x, nthreads_y, nthreads_z);
  const dim3 nblocks(nblocks_x, nblocks_y, nblocks_z);
  printf("blur_image\n");
  blur_image_kernel<<<nblocks, nthreads>>>();
  // copy to host
  cudaDeviceSynchronize();

  return std::vector<float>();
}

int main() {
  printf("main\n");
  // initialize image (ideally as 2D vector?)
  // const std::vector< std::vector<float> > image{ {0, 1},
  //                                                {2, 3} };
  constexpr size_t nrows{3};
  constexpr size_t ncols{3};
  std::vector<float> image{ 0, 0, 0,
                            0, 1, 0,
                            0, 0, 0,
  };
  if (nrows * ncols != image.size()) {
    throw std::runtime_error("Something is wrong");
  }
  const auto blurred_image = blur_image(image.data(), nrows, ncols);
  return 0;
}
