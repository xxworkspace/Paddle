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

#include "llvm/IR/Module.h"
#include "paddle/fluid/compiler/piano/backends/compiler.h"

namespace paddle {
namespace piano {
namespace backends {

// LlvmCompiler is an abstract compiler class that inherit 'Compiler' with
// llvm ir as low level IR
// For a particular device compiler with llvm ir should inherit the LlvmCompiler
// and overwrite the function 'Optimize' and 'Compile'

class LlvmCompiler : public Compiler {
 public:
  LlvmCompiler() = default;
  virtual ~LlvmCompiler() {}

  KernelExecutors Apply(std::unique_ptr<note::Module>&) override;

 protected:
  virtual void Optimize(std::unique_ptr<note::Module>&) = 0;
  void ConvertToIr(const std::unique_ptr<note::Module>&,
                   std::unique_ptr<llvm::Module>&, KernelExecutors&);
  virtual void Compile(std::unique_ptr<llvm::Module>&, KernelExecutors&) = 0;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
