/******************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "amo_bitwise_tester.hpp"

#include <iostream>
#include <roc_shmem/roc_shmem.hpp>

using namespace rocshmem;

/* Declare the global kernel template with a generic implementation */
template <typename T>
__global__ void AMOBitwiseTest(int loop, int skip, uint64_t *timer, char *r_buf,
                               T *s_buf, T *ret_val, TestType type,
                               ShmemContextType ctx_type) {
  return;
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
template <typename T>
AMOBitwiseTester<T>::AMOBitwiseTester(TesterArguments args) : Tester(args) {
  hipMalloc((void **)&_ret_val, args.max_msg_size * args.num_wgs);
  _r_buf = (char *)roc_shmem_malloc(args.max_msg_size);
  _s_buf = (T *)roc_shmem_malloc(args.max_msg_size * args.num_wgs);
}

template <typename T>
AMOBitwiseTester<T>::~AMOBitwiseTester() {
  roc_shmem_free(_r_buf);
  hipFree(_ret_val);
}

template <typename T>
void AMOBitwiseTester<T>::resetBuffers(uint64_t size) {
  memset(_r_buf, 0, args.max_msg_size);
  memset(_ret_val, 0, args.max_msg_size * args.num_wgs);
  memset(_s_buf, 0, args.max_msg_size * args.num_wgs);
}

template <typename T>
void AMOBitwiseTester<T>::launchKernel(dim3 gridsize, dim3 blocksize, int loop,
                                       uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(AMOBitwiseTest, gridsize, blocksize, shared_bytes, stream,
                     loop, args.skip, timer, _r_buf, _s_buf, _ret_val, _type,
                     _shmem_context);

  _gridSize = gridsize;
  num_msgs = (loop + args.skip) * gridsize.x;
  num_timed_msgs = loop;
}

template <typename T>
void AMOBitwiseTester<T>::verifyResults(uint64_t size) {
  T ret;
  if (args.myid == 0) {
    T expected_val = 0;

    switch (_type) {
      case AMO_FetchAndTestType:
        expected_val = 0;
        break;
      case AMO_AndTestType:
        expected_val = 0;
        break;
      case AMO_FetchOrTestType:
        expected_val = 0xFFFF;
        break;
      case AMO_OrTestType:
        expected_val = 0xFFFF;
        break;
      case AMO_FetchXorTestType:
        expected_val = 0xFFFF;
        break;
      case AMO_XorTestType:
        expected_val = (num_msgs % 2) ? 0xFFFF : 0;
        break;
      default:
        break;
    }

    int fetch_op =
        (_type == AMO_FetchAndTestType || _type == AMO_FetchOrTestType ||
         _type == AMO_FetchXorTestType)
            ? 1
            : 0;

    if (fetch_op == 1) {
      ret = *std::max_element(_ret_val, _ret_val + args.num_wgs);
    } else {
      ret = *std::max_element(_s_buf, _s_buf + args.num_wgs);
    }
    if (ret != expected_val) {
      std::cerr << "data validation error\n";
      std::cerr << "got " << ret << ", expected " << expected_val << std::endl;
      exit(-1);
    }
  }
}

#define AMO_BITWISE_DEF_GEN(T, TNAME)                                         \
  template <>                                                                 \
  __global__ void AMOBitwiseTest<T>(                                          \
      int loop, int skip, uint64_t *timer, char *r_buf, T *s_buf, T *ret_val, \
      TestType type, ShmemContextType ctx_type) {                             \
    __shared__ roc_shmem_ctx_t ctx;                                           \
    roc_shmem_wg_init();                                                      \
    roc_shmem_wg_ctx_create(ctx_type, &ctx);                                  \
    if (hipThreadIdx_x == 0) {                                                \
      uint64_t start;                                                         \
      T ret = 0;                                                              \
      T cond = 0;                                                             \
      for (int i = 0; i < loop + skip; i++) {                                 \
        if (i == skip) start = roc_shmem_timer();                             \
        switch (type) {                                                       \
          case AMO_FetchAndTestType:                                          \
            ret = roc_shmem_ctx_##TNAME##_atomic_fetch_and(ctx, (T *)r_buf,   \
                                                           0xFFFF, 1);        \
            break;                                                            \
          case AMO_AndTestType:                                               \
            roc_shmem_ctx_##TNAME##_atomic_and(ctx, (T *)r_buf, 0xFFFF, 1);   \
            break;                                                            \
          case AMO_FetchOrTestType:                                           \
            ret = roc_shmem_ctx_##TNAME##_atomic_fetch_or(ctx, (T *)r_buf,    \
                                                          0xFFFF, 1);         \
            break;                                                            \
          case AMO_OrTestType:                                                \
            roc_shmem_ctx_##TNAME##_atomic_or(ctx, (T *)r_buf, 0xFFFF, 1);    \
            break;                                                            \
          case AMO_FetchXorTestType:                                          \
            ret = roc_shmem_ctx_##TNAME##_atomic_fetch_xor(ctx, (T *)r_buf,   \
                                                           0xFFFF, 1);        \
            break;                                                            \
          case AMO_XorTestType:                                               \
            roc_shmem_ctx_##TNAME##_atomic_xor(ctx, (T *)r_buf, 0xFFFF, 1);   \
            break;                                                            \
          default:                                                            \
            break;                                                            \
        }                                                                     \
      }                                                                       \
      roc_shmem_ctx_quiet(ctx);                                               \
      timer[hipBlockIdx_x] = roc_shmem_timer() - start;                       \
      ret_val[hipBlockIdx_x] = ret;                                           \
      roc_shmem_ctx_getmem(ctx, &s_buf[hipBlockIdx_x], r_buf, sizeof(T), 1);  \
    }                                                                         \
    roc_shmem_wg_ctx_destroy(&ctx);                                            \
    roc_shmem_wg_finalize();                                                  \
  }                                                                           \
  template class AMOBitwiseTester<T>;

AMO_BITWISE_DEF_GEN(unsigned int, uint)
AMO_BITWISE_DEF_GEN(unsigned long, ulong)
AMO_BITWISE_DEF_GEN(unsigned long long, ulonglong)
