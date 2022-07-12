#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>

#include <iostream>
#include <random>
#include <set>

int main() {
  thrust::minstd_rand rng(std::random_device{}());
  thrust::uniform_real_distribution<float> dist(-7, 13);

  std::set<float> s;
  for (int i = 0; i < 100; ++i) {
    auto f = dist(rng);
    std::cout << f << " ";
    if (i % 10 == 9) std::cout << "\n";
    s.insert(f);
  }
  std::cout << "\n";
  std::cout << s.size() << "\n";

  return 0;
}
