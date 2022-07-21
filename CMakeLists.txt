cmake_minimum_required(VERSION 3.20.1)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 75)
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --extended-lambda --expt-relaxed-constexpr)

project(CudaTopKBench CUDA)
find_package(CUDAToolkit REQUIRED)

add_subdirectory(nvbench)

add_executable(main_bench main_bench.cu topk.cuh yxz_topk.cu yxz_topk_heap.cu anil_bitonic_topk.cu)
target_link_libraries(main_bench nvbench::main CUDA::curand)
# target_compile_options(main_bench PUBLIC
#     $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>) 

# target_compile_options(main_bench "--extended-lambda")
