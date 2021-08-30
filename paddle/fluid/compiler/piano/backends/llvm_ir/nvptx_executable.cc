// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_executable.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/executable_pool.h"

namespace paddle {
namespace piano {
namespace backends {

void NvtpxExecutable::Launch(std::vector<void*>& args, void* stream) {
  // get CUfunction name_ = func_name
  auto func = CumodulePool::Instance().GetCuFunction(module_name_, name_);
  CHECK_CUDA_DRIVER_SUCCESS(cuLaunchKernel(
      func, grid_dim_.x, grid_dim_.y, grid_dim_.x, block_dim_.x, block_dim_.y,
      block_dim_.z, 0, static_cast<CUstream>(stream), args.data(), nullptr));
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
