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

#include "team_ctx_primitive_tester.hpp"

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void TeamCtxPrimitiveTest(int loop, int skip, uint64_t *start_time,
                                     uint64_t *end_time, char *s_buf,
                                     char *r_buf, int size, TestType type,
                                     ShmemContextType ctx_type,
                                     rocshmem_team_t *teams) {
  __shared__ rocshmem_ctx_t ctx;
  int wg_id = get_flat_grid_id();
  rocshmem_wg_init();
  rocshmem_wg_team_create_ctx(teams[wg_id], ctx_type, &ctx);

  
  // Calculate start index for each thread within the grid
  uint64_t idx = size * get_flat_id();
  s_buf += idx;
  r_buf += idx;

  for (int i = 0; i < loop + skip; i++) {
    if (i == skip) {
        __syncthreads();
        start_time[wg_id] = wall_clock64();
    }

    switch (type) {
      case TeamCtxGetTestType:
        rocshmem_ctx_getmem(ctx, r_buf, s_buf, size, 1);
        break;
      case TeamCtxGetNBITestType:
        rocshmem_ctx_getmem_nbi(ctx, r_buf, s_buf, size, 1);
        break;
      case TeamCtxPutTestType:
        rocshmem_ctx_putmem(ctx, r_buf, s_buf, size, 1);
        break;
      case TeamCtxPutNBITestType:
        rocshmem_ctx_putmem_nbi(ctx, r_buf, s_buf, size, 1);
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
TeamCtxPrimitiveTester::TeamCtxPrimitiveTester(TesterArguments args)
    : Tester(args) {
  size_t buff_size = args.max_msg_size * args.wg_size * args.num_wgs;
  s_buf = (char *)rocshmem_malloc(buff_size);
  r_buf = (char *)rocshmem_malloc(buff_size);

  char* value{nullptr};
  if ((value = getenv("ROCSHMEM_MAX_NUM_TEAMS"))) {
    num_teams = atoi(value);
  }

  CHECK_HIP(hipMalloc(&team_primitive_world_dup,
                      sizeof(rocshmem_team_t) * num_teams));
}

TeamCtxPrimitiveTester::~TeamCtxPrimitiveTester() {
  rocshmem_free(s_buf);
  rocshmem_free(r_buf);
  CHECK_HIP(hipFree(team_primitive_world_dup));
}

void TeamCtxPrimitiveTester::resetBuffers(uint64_t size) {
  size_t buff_size = size * args.wg_size * args.num_wgs;
  memset(s_buf, '0', buff_size);
  memset(r_buf, '1', buff_size);
}

void TeamCtxPrimitiveTester::preLaunchKernel() {
  int n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  for (int team_i = 0; team_i < num_teams; team_i++) {
    team_primitive_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                 &team_primitive_world_dup[team_i]);
    if (team_primitive_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      printf("Team %d is invalid!\n", team_i);
      abort();
    }
  }
}

void TeamCtxPrimitiveTester::launchKernel(dim3 gridSize, dim3 blockSize,
                                          int loop, uint64_t size) {
  size_t shared_bytes = 0;

  hipLaunchKernelGGL(TeamCtxPrimitiveTest, gridSize, blockSize,
                     shared_bytes, stream, loop, args.skip, start_time,
                     end_time, s_buf, r_buf, size, _type,
                     _shmem_context, team_primitive_world_dup);

  num_msgs = (loop + args.skip) * gridSize.x * blockSize.x;
  num_timed_msgs = loop * gridSize.x * blockSize.x;
}

void TeamCtxPrimitiveTester::postLaunchKernel() {
  for (int team_i = 0; team_i < num_teams; team_i++) {
    rocshmem_team_destroy(team_primitive_world_dup[team_i]);
  }
}

void TeamCtxPrimitiveTester::verifyResults(uint64_t size) {
  int check_id =
      (_type == TeamCtxGetTestType || _type == TeamCtxGetNBITestType) ? 0 : 1;

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
