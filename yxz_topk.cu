// data type
#define DATATYPE float

// min number
#define NEG_INF -999999999

// use gpu
#define USE_GPU

#ifdef USE_GPU
#define HANDLE_CUDA_ERROR(err) (handleCudaError(err, __FILE__, __LINE__))
#define GPU_BLOCKS_THRESHOLD 2048
#define GPU_THREADS_THRESHOLD 1024
#define GPU_SHARED_MEM_THRESHOLD 48 * 1024
#define GPU_THREADS 128
#endif

// cuda error
void handleCudaError(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(2);
  }
}

__device__ inline void replace_smaller(DATATYPE* array, int k, DATATYPE data) {
  if (data < array[k - 1]) return;
  for (int j = k - 2; j >= 0; j--) {
    if (data > array[j])
      array[j + 1] = array[j];
    else {
      array[j + 1] = data;
      return;
    }
  }
  array[0] = data;
}

__device__ inline void mergeTwoK(DATATYPE* left, DATATYPE* right, int k) {
  int i;
  for (i = 0; i < k; i++) {
    replace_smaller(left, k, right[i]);
  }
}

__global__ void top_k_gpu_kernel3_1(DATATYPE* input, int length, int k,
                                    DATATYPE* output) {
  extern __shared__ DATATYPE shared_buffer[];
  DATATYPE* myPoint = shared_buffer + threadIdx.x * k;
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int threadNum = gridDim.x * blockDim.x;
  int localThreadId = threadIdx.x;
  int localThreadNum = blockDim.x;
  int i, index;

  for (index = 0, i = blockIdx.x * localThreadNum * k + localThreadId;
       index < k; index++, i += localThreadNum) {
    myPoint[index] = NEG_INF;
    replace_smaller(myPoint, index + 1, input[i]);
  }
  // replace the data if bigger
  for (i = k * threadNum + threadId; i < length; i += threadNum) {
    replace_smaller(myPoint, k, input[i]);
  }
  __syncthreads();

  // reduction
  for (i = localThreadNum >> 1; i > 0; i >>= 1) {
    if (localThreadId < i) {
      mergeTwoK(myPoint, myPoint + i * k, k);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // produce k data in decent order
    index = blockIdx.x * localThreadNum * k;
    for (i = 0; i < k; i++) {
      input[index + i] = myPoint[i];
    }
  }
}

__global__ void top_k_gpu_kernel3_2(DATATYPE* input, int num, int stride, int k,
                                    DATATYPE* output) {
  extern __shared__ DATATYPE shared_buffer[];
  DATATYPE* myPoint = shared_buffer + threadIdx.x * k;
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int threadNum = gridDim.x * blockDim.x;
  int localThreadId = threadIdx.x;
  int localThreadNum = blockDim.x;
  int i;

  for (i = 0; i < k; i++) myPoint[i] = input[threadId * stride + i];
  for (i = threadNum + threadId; i < num; i += threadNum)
    mergeTwoK(myPoint, input + i * stride, k);
  __syncthreads();

  // reduction
  for (i = localThreadNum >> 1; i > 0; i >>= 1) {
    if (localThreadId < i) {
      mergeTwoK(myPoint, myPoint + i * k, k);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // produce k data in decent order
    DATATYPE* outputPoint = output + blockIdx.x * localThreadNum * stride;
    for (i = 0; i < k; i++) {
      outputPoint[i] = myPoint[i];
    }
  }
}

__global__ void top_k_gpu_kernel3_1_orig(DATATYPE* input, int length, int k,
                                         DATATYPE* output) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int threadNum = gridDim.x * blockDim.x;
  int localThreadId = threadIdx.x;
  int localThreadNum = blockDim.x;
  DATATYPE* myPoint = input + threadId * k;
  int i, index;

  for (index = 0; index < k; index++) {
    replace_smaller(myPoint, index + 1, myPoint[index]);
  }
  // replace the data if bigger
  for (i = k * threadNum + threadId; i < length; i += threadNum) {
    replace_smaller(myPoint, k, input[i]);
  }
  __syncthreads();

  // reduction
  for (i = localThreadNum >> 1; i > 0; i >>= 1) {
    if (localThreadId < i) {
      mergeTwoK(myPoint, myPoint + i * k, k);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // produce k data in decent order
    DATATYPE* outputPoint = input + blockIdx.x * localThreadNum * k;
    for (i = 0; i < k; i++) {
      outputPoint[i] = myPoint[i];
    }
  }
}
__global__ void top_k_gpu_kernel3_2_orig(DATATYPE* input, int num, int stride,
                                         int k, DATATYPE* output) {
  int threadId = blockIdx.x * blockDim.x + threadIdx.x;
  int threadNum = gridDim.x * blockDim.x;
  int localThreadId = threadIdx.x;
  int localThreadNum = blockDim.x;
  int i;
  DATATYPE* myPoint = input + threadId * stride;

  for (i = threadNum + threadId; i < num; i += threadNum)
    mergeTwoK(myPoint, input + i * stride, k);
  __syncthreads();

  // reduction
  for (i = localThreadNum >> 1; i > 0; i >>= 1) {
    if (localThreadId < i) {
      mergeTwoK(myPoint, myPoint + i * stride, k);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    // produce k data in decent order
    DATATYPE* outputPoint = output + blockIdx.x * localThreadNum * stride;
    for (i = 0; i < k; i++) {
      outputPoint[i] = myPoint[i];
    }
  }
}

void yxz_topk(DATATYPE* input, int length, int k, DATATYPE* output,
              cudaStream_t stream = 0) {
  // each thread at least 2K
  int blocks_opt, thread_opt;
  if (k < 20) {
    blocks_opt = GPU_BLOCKS_THRESHOLD;
    thread_opt = GPU_THREADS;
  } else {
    blocks_opt = 46 * 16;
    thread_opt = 64;
  }
  int threads =
      (thread_opt < length / (4 * k) * 2) ? thread_opt : (length / (4 * k) * 2);
  int stride = threads * k;
  int blocks = (blocks_opt < length / (threads * 2 * k))
                   ? blocks_opt
                   : (length / (threads * 2 * k));
  int shared_mem_usage = sizeof(DATATYPE) * k * threads;
  // printf("shared mem usage: (%d %d) %d(%d)\n", blocks, threads,
  // shared_mem_usage, GPU_SHARED_MEM_THRESHOLD);
  if (shared_mem_usage < GPU_SHARED_MEM_THRESHOLD)
    top_k_gpu_kernel3_1<<<blocks, threads, shared_mem_usage, stream>>>(
        input, length, k, output);
  else
    top_k_gpu_kernel3_1_orig<<<blocks, threads, 0, stream>>>(input, length, k,
                                                             output);
  threads = (thread_opt < blocks / 2) ? thread_opt : (blocks / 2);
  shared_mem_usage = sizeof(DATATYPE) * k * threads;
  // printf("shared mem usage: (%d %d) %d(%d)\n", 1, threads, shared_mem_usage,
  // GPU_SHARED_MEM_THRESHOLD);
  if (shared_mem_usage < GPU_SHARED_MEM_THRESHOLD)
    top_k_gpu_kernel3_2<<<1, threads, shared_mem_usage, stream>>>(
        input, blocks, stride, k, output);
  else
    top_k_gpu_kernel3_2_orig<<<1, threads, 0, stream>>>(input, blocks, stride,
                                                        k, output);
  cudaError_t err = cudaGetLastError();
  HANDLE_CUDA_ERROR(err);
}
