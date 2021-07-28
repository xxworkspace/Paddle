// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "llvm/IR/Module.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/gpu/gpu_compiler.h"

namespace piano {
namespace gpu {

class NvptxCompiler : public GpuCompiler{
protected:
    Status Optimize(std::unique_ptr<NoteModule>&) override;
    std::unique_ptr<Executable> Compile(std::unique_ptr<llvm::Module>&, Schedules*) override;

    Status InitNvptxContext();
    Status OptimizeLlvmIR(std::unique_ptr<llvm::Module>&);
    std::string ConverToPtx(std::unique_ptr<llvm::Module>&);

    std::string GetLlvmTarget();
    std::string GetLlvmDataLayout();
};

}
}
