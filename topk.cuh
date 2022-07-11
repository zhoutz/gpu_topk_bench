void yxz_topk(float* input, int length, int k, float* output,
              cudaStream_t stream);

void yxz_topk_heap(float* input, int length, int k, float* output,
                   cudaStream_t stream);
