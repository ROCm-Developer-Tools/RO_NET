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

#include "primitive_tester.hpp"

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void PrimitiveTest(int loop, int skip, uint64_t *start_time,
                              uint64_t *end_time, char *s_buf, char *r_buf,
                              int size, TestType type,
                              ShmemContextType ctx_type) {
  __shared__ rocshmem_ctx_t ctx;
  rocshmem_wg_init();
  rocshmem_wg_ctx_create(ctx_type, &ctx);
  int wg_id = get_flat_grid_id();

  /**
   * Calculate start index for each thread within the grid
   */
  uint64_t idx = size * get_flat_id();
  s_buf += idx;
  r_buf += idx;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
        __syncthreads();
        start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case GetTestType:
        rocshmem_ctx_getmem(ctx, r_buf, s_buf, size, 1);
        break;
      case GetNBITestType:
        rocshmem_ctx_getmem_nbi(ctx, r_buf, s_buf, size, 1);
        break;
      case PutTestType:
        rocshmem_ctx_putmem(ctx, r_buf, s_buf, size, 1);
        break;
      case PutNBITestType:
        rocshmem_ctx_putmem_nbi(ctx, r_buf, s_buf, size, 1);
        break;
      case PTestType:
        for (int s = 0; s < size; s++) {
          char val = s_buf[s];
          rocshmem_ctx_char_p(ctx, &r_buf[s], val, 1);
        }
        break;
      case GTestType:
        for (int s = 0; s < size; s++) {
          char ret = rocshmem_ctx_char_g(ctx, &s_buf[s], 1);
          r_buf[s] = ret;
        }
        break;
      default:
        break;
    }
  }

  rocshmem_ctx_quiet(ctx);

  __syncthreads();

  if (hipThreadIdx_x == 0) {
    end_time[wg_id] = wall_clock64();
  }

  rocshmem_wg_ctx_destroy(&ctx);
  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
PrimitiveTester::PrimitiveTester(TesterArguments args) : Tester(args) {
  size_t buff_size = args.max_msg_size * args.wg_size * args.num_wgs;
  s_buf = (char *)rocshmem_malloc(buff_size);
  r_buf = (char *)rocshmem_malloc(buff_size);
}

PrimitiveTester::~PrimitiveTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
}

void PrimitiveTester::resetBuffers(uint64_t size) {
  size_t buff_size = size * args.wg_size * args.num_wgs;
  memset(s_buf, '0', buff_size);
  memset(r_buf, '1', buff_size);
}

void PrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                   uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(PrimitiveTest, gridSize, blockSize, shared_bytes, stream,
                     loop, args.skip, start_time, end_time, s_buf, r_buf,
                     size, _type, _shmem_context);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void PrimitiveTester::verifyResults(uint64_t size) {
  int check_id =
      (_type == GetTestType || _type == GetNBITestType || _type == GTestType)
          ? 0
          : 1;

  if (args.myid == check_id) {
    size_t buff_size = size * args.wg_size * args.num_wgs;
    for (uint64_t i = 0; i < buff_size; i++) {
      if (r_buf[i] != '0') {
        fprintf(stderr, "Data validation error at idx %lu\n", i);
        fprintf(stderr, "Got %c, Expected %c\n", r_buf[i], '0');
        exit(-1);
      }
    }
  }
}
