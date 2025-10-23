g++ store_matrix_example.cpp -o store_matrix_example.out
nvcc vector_add_dim3.cu -o vector_add_dim3.out -arch=sm_89
nvcc blur_image.cu -o blur_image.out -arch=sm_89
