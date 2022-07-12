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
#include <random>

#include "topk.cuh"

constexpr int N = 2e7;

struct GPU_RNG {
  thrust::random::taus88 rng;
  thrust::random::uniform_real_distribution<float> dist;
  GPU_RNG() : rng(std::random_device{}()), dist(0.f, 1.f) {}
  __device__ float operator()() { return dist(rng); }
} rng;

void generate_random_device_vector(thrust::device_vector<float> &v) {
  thrust::generate(thrust::device, v.begin(), v.end(), rng);
}

struct TopK_Benchmark {
  using test_func_t = void (*)(float *, int, int, float *, cudaStream_t);
  test_func_t p;
  TopK_Benchmark(test_func_t p) : p(p) {}

  void operator()(nvbench::state &state) {
    const auto k = state.get_int64("k");
    auto src_vec = thrust::device_vector<float>(N);
    auto result = thrust::device_vector<float>(k);
    // state.add_global_memory_reads<nvbench::int32_t>(N, "DataSize");
    state.exec(
        nvbench::exec_tag::timer,
        [k, &src_vec, &result, this](nvbench::launch &launch, auto &timer) {
          generate_random_device_vector(src_vec);
          timer.start();
          this->p(thrust::raw_pointer_cast(src_vec.data()), N, k,
                  thrust::raw_pointer_cast(result.data()), launch.get_stream());
          timer.stop();
#if 0
      thrust::sort(src_vec.begin(), src_vec.end(), thrust::greater<float>());
      thrust::sort(result.begin(), result.end(), thrust::greater<float>());
      for (int i = 0; i < k; i++) {
        if (result[i] != src_vec[i]) {
          printf("i = %d: %f != %f\n", i, float(result[i]), float(src_vec[i]));
        }
      }
#endif
        });
  }
};

struct TopK_Benchmark_WithBuffer {
  using test_func_t = void (*)(float *, int, int, float *, cudaStream_t,
                               float *);
  test_func_t p;
  TopK_Benchmark_WithBuffer(test_func_t p) : p(p) {}

  void operator()(nvbench::state &state) {
    const auto k = state.get_int64("k");
    auto src_vec = thrust::device_vector<float>(N);
    auto buffer = thrust::device_vector<float>(N);
    auto result = thrust::device_vector<float>(k);
    // state.add_global_memory_reads<nvbench::int32_t>(N, "DataSize");
    state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
               [k, &src_vec, &result, &buffer, this](nvbench::launch &launch,
                                                     auto &timer) {
                 generate_random_device_vector(src_vec);
                 timer.start();
                 this->p(thrust::raw_pointer_cast(src_vec.data()), N, k,
                         thrust::raw_pointer_cast(result.data()),
                         launch.get_stream(),
                         thrust::raw_pointer_cast(buffer.data()));
                 timer.stop();
#if 0
      thrust::sort(src_vec.begin(), src_vec.end(), thrust::greater<float>());
      thrust::sort(result.begin(), result.end(), thrust::greater<float>());
      for (int i = 0; i < k; i++) {
        if (result[i] != src_vec[i]) {
          printf("i = %d: %f != %f\n", i, float(result[i]), float(src_vec[i]));
        }
      }
#endif
               });
  }
};

auto yxz_topk_benchmark = TopK_Benchmark(yxz_topk);
auto yxz_topk_heap_benchmark = TopK_Benchmark(yxz_topk_heap);
auto anil_bitonic_benchmark = TopK_Benchmark_WithBuffer(anil_bitonic<float>);

constexpr int range_max = 48;

NVBENCH_BENCH(yxz_topk_benchmark)
    .add_int64_axis("k", nvbench::range(1, range_max));
NVBENCH_BENCH(yxz_topk_heap_benchmark)
    .add_int64_axis("k", nvbench::range(1, range_max));
NVBENCH_BENCH(anil_bitonic_benchmark)
    .add_int64_axis("k", nvbench::range(1, range_max));
