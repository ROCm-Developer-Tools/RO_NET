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

#include "extended_primitives.hpp"

#include <rocshmem/rocshmem.hpp>

#include <numeric>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void ExtendedPrimitiveTest(int loop, int skip, uint64_t *start_time,
                                      uint64_t *end_time, char *s_buf,
                                      char *r_buf, int size, TestType type,
                                      ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_init();
  rocshmem_wg_ctx_create(ctx_type, &ctx);

  
  // Calculate start index for each work group
  int wg_id = get_flat_grid_id();
  uint64_t offset = size * wg_id;
  s_buf += offset;
  r_buf += offset;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
        start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case WGGetTestType:
        rocshmem_ctx_getmem_wg(ctx, r_buf, s_buf, size, 1);
        break;
      case WGGetNBITestType:
        rocshmem_ctx_getmem_nbi_wg(ctx, r_buf, s_buf, size, 1);
        break;
      case WGPutTestType:
        rocshmem_ctx_putmem_wg(ctx, r_buf, s_buf, size, 1);
        break;
      case WGPutNBITestType:
        rocshmem_ctx_putmem_nbi_wg(ctx, r_buf, s_buf, size, 1);
        break;
      default:
        break;
    }
  }

  rocshmem_ctx_quiet(ctx);

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
ExtendedPrimitiveTester::ExtendedPrimitiveTester(TesterArguments args)
    : Tester(args) {
  s_buf = static_cast<int*>(rocshmem_malloc(args.max_msg_size * args.num_wgs));
  r_buf = static_cast<int*>(rocshmem_malloc(args.max_msg_size * args.num_wgs));
}

ExtendedPrimitiveTester::~ExtendedPrimitiveTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
}

void ExtendedPrimitiveTester::resetBuffers(uint64_t size) {
  num_elems = (size * args.num_wgs) / sizeof(int);
  std::iota(s_buf, s_buf + num_elems, 0);
  memset(r_buf, 0, size * args.num_wgs);
}

void ExtendedPrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                           int loop, uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(ExtendedPrimitiveTest, gridSize, blockSize, shared_bytes,
                     stream, loop, args.skip, start_time, end_time,
                     (char*)s_buf, (char*)r_buf, size, _type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x;
  num_timed_msgs = loop * gridSize.x;
}

void ExtendedPrimitiveTester::verifyResults(uint64_t size) {
  int check_id = (_type == WGGetTestType || _type == WGGetNBITestType)
                     ? 0
                     : 1;

  if (args.myid == check_id) {
    for (int i = 0; i < num_elems; i++) {
      if (r_buf[i] != i) {
        fprintf(stderr, "Data validation error at idx %d\n", i);
        fprintf(stderr, "Got %d, Expected %d \n", r_buf[i], i);
        exit(-1);
      }
    }
  }
}
