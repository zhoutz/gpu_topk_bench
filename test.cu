#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/uniform_real_distribution.h>
int main(void) {
  // create a minstd_rand object to act as our source of randomness
  thrust::minstd_rand rng;
  // create a uniform_real_distribution to produce floats from [-7,13)
  thrust::uniform_real_distribution<float> dist(-7, 13);
  // write a random number from the range [-7,13) to standard output
  std::cout << dist(rng) << std::endl;
  // write the range of the distribution, just in case we forgot
  std::cout << dist.min() << std::endl;
  // -7.0 is printed
  std::cout << dist.max() << std::endl;
  // 13.0 is printed
  // write the parameters of the distribution (which happen to be the bounds) to
  // standard output
  std::cout << dist.a() << std::endl;
  // -7.0 is printed
  std::cout << dist.b() << std::endl;
  // 13.0 is printed
  return 0;
}
