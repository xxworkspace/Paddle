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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_compiler.h"
#include "llvm/Support/TargetSelect.h"

namespace paddle {
namespace piano {
namespace backends {

void NvptxCompiler::Optimize(std::unique_ptr<note::Module>& note_module) {}

void NvptxCompiler::Compile(std::unique_ptr<llvm::Module>& llvm_module,
                            KernelExecutors& kernel_executors_) {}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
