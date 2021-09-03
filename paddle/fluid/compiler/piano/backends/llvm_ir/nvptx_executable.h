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
#pragma once

#include <cuda_runtime_api.h>
#include "paddle/fluid/compiler/piano/backends/kernel_executable.h"

namespace paddle {
namespace piano {
namespace backends {

// Nvptx Executable for nvidia jit execution.
// TODO(sunli) : overwrite the function Run.
class NvptxExecutable : public KernelExecutable {
 public:
  NvptxExecutable(const std::string& module_name, const dim3& grid_dim,
                  const dim3& block_dim, const uint32_t shared_size,
                  const note::Instruction& note_instruction)
      : KernelExecutable(note_instruction) {
    module_name_ = module_name;
    grid_dim_ = grid_dim;
    block_dim_ = block_dim;
    shared_size_ = shared_size;
  }
  void Launch(std::vector<void*>&, void*) override;

 private:
  std::string module_name_;
  dim3 grid_dim_;
  dim3 block_dim_;
  uint32_t shared_size_;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
