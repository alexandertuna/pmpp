#include <iostream>

int main() {

  int device_count{0};
  cudaGetDeviceCount(&device_count);
  printf("Device count: %i\n", device_count);

  for (int dev{0}; dev < device_count; ++dev) {
    cudaDeviceProp prop;
    const auto err = cudaGetDeviceProperties(&prop, dev);
    printf("Device %i: %s has %i sharedMemPerBlock\n", dev, prop.name, prop.sharedMemPerBlock);
  }
  
  printf("Getting device props ...\n");
  return 0;
}
