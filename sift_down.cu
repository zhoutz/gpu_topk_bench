template <class _Compare, class T>
__device__ __host__ void sift_down(T* __first, int __len, T* __start,
                                   _Compare __comp) {
  // left-child of __start is at 2 * __start + 1
  // right-child of __start is at 2 * __start + 2
  int __child = __start - __first;

  if (__len < 2 || (__len - 2) / 2 < __child) return;

  __child = 2 * __child + 1;
  T* __child_i = __first + __child;

  if ((__child + 1) < __len && __comp(*__child_i, *(__child_i + 1))) {
    // right-child exists and is greater than left-child
    ++__child_i;
    ++__child;
  }

  // check if we are in heap-order
  if (__comp(*__child_i, *__start))
    // we are, __start is larger than it's largest child
    return;

  T __top((*__start));
  do {
    // we are not in heap-order, swap the parent with its largest child
    *__start = (*__child_i);
    __start = __child_i;

    if ((__len - 2) / 2 < __child) break;

    // recompute the child based off of the updated parent
    __child = 2 * __child + 1;
    __child_i = __first + __child;

    if ((__child + 1) < __len && __comp(*__child_i, *(__child_i + 1))) {
      // right-child exists and is greater than left-child
      ++__child_i;
      ++__child;
    }

    // check if we are in heap-order
  } while (!__comp(*__child_i, __top));
  *__start = (__top);
}

template <class _Compare, class T>
__device__ __host__ void make_heap(T* __first, T* __last, _Compare __comp) {
  int __n = __last - __first;
  if (__n > 1) {
    // start from the first parent, there is no need to consider children
    for (int __start = (__n - 2) / 2; __start >= 0; --__start) {
      sift_down(__first, __n, __first + __start, __comp);
    }
  }
}
