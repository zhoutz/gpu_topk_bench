#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#include <cuda/std/chrono>
#include <nvbench/nvbench.cuh>
// #include <cuda_runtime.h>

#include <random>

#include "yxz_topk_heap.cu"
// #include "yxz_topk.cu"

// __global__ void sleep_kernel(nvbench::int64_t microseconds) {
//   const auto start = cuda::std::chrono::high_resolution_clock::now();
//   const auto target_duration = cuda::std::chrono::microseconds(microseconds);
//   const auto finish = start + target_duration;

//   auto now = cuda::std::chrono::high_resolution_clock::now();
//   while (now < finish) {
//     now = cuda::std::chrono::high_resolution_clock::now();
//   }
// }
// void sleep_benchmark(nvbench::state &state) {
//   const auto duration_us = state.get_int64("Duration (us)");
//   state.exec([&duration_us](nvbench::launch &launch) {
//     sleep_kernel<<<1, 1, 0, launch.get_stream()>>>(duration_us);
//   });
// }
// NVBENCH_BENCH(sleep_benchmark)
//     .add_int64_axis("Duration (us)", nvbench::range(0, 100, 5))
//     .set_timeout(1); // Limit to one second per measurement.

constexpr int N = 1e7;

struct GPU_RNG {
  thrust::random::minstd_rand rng;
  thrust::random::uniform_real_distribution<float> dist;
  GPU_RNG() : rng(std::random_device{}()), dist(0, 1) {}
  __device__ float operator()() { return dist(rng); }
} rng;

void generate_random_vector(thrust::device_vector<float> &v) {
  thrust::generate(thrust::device, v.begin(), v.end(), rng);
}

void yxz_benchmark(nvbench::state &state) {
  const int k = state.get_int64("k");
  auto src_vec = thrust::device_vector<float>(N);
  auto result = thrust::device_vector<float>(k);
  state.add_global_memory_reads<nvbench::int32_t>(N, "DataSize");
  state.exec(nvbench::exec_tag::timer, [k, &src_vec, &result](
                                           nvbench::launch &launch,
                                           auto &timer) {
    generate_random_vector(src_vec);
    timer.start();
    yxz_topk(thrust::raw_pointer_cast(src_vec.data()), src_vec.size(), k,
             thrust::raw_pointer_cast(result.data()), launch.get_stream());
    timer.stop();
    thrust::sort(src_vec.begin(), src_vec.end(), thrust::greater<float>());
    thrust::sort(result.begin(), result.end(), thrust::greater<float>());
    for (int i = 0; i < k; i++) {
      if (result[i] != src_vec[i]) {
        printf("i = %d: %f != %f\n", i, float(result[i]), float(src_vec[i]));
      }
    }
  });
}
NVBENCH_BENCH(yxz_benchmark).add_int64_axis("k", nvbench::range(1, 41))
    // .set_timeout(1)  // Limit to one second per measurement.
    ;
