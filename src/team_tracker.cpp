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

#include "team_tracker.hpp"

#include <cstdlib>

namespace rocshmem {

TeamTracker::TeamTracker() {
  char* value{nullptr};
  if ((value = getenv("ROC_SHMEM_MAX_NUM_TEAMS"))) {
    max_num_teams_ = atoi(value);
  }
}

void TeamTracker::track(roc_shmem_team_t team) {
  if (team == ROC_SHMEM_TEAM_INVALID) {
    return;
  }
  teams_.push_back(team);
}

void TeamTracker::untrack(roc_shmem_team_t team) {
  auto it{std::find(teams_.begin(), teams_.end(), team)};
  assert(it != teams_.end());
  teams_.erase(it);
}

}  // namespace rocshmem
