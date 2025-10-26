#include <iostream>

int main() {
  int dev_count;
  cudaGetDeviceCount(&dev_count);
  printf("Device count: %i\n", dev_count);
}
