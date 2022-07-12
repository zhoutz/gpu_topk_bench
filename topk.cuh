void yxz_topk(float* input, int length, int k, float* output,
              cudaStream_t stream);

void yxz_topk_heap(float* input, int length, int k, float* output,
                   cudaStream_t stream);

template <typename KeyT>
void anil_bitonic(KeyT* d_keys_in, int num_items, int k, KeyT* d_keys_out,
                  cudaStream_t stream, KeyT* d_temp);
