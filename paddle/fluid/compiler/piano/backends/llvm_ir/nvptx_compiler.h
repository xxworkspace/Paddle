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

#include <mutex>
#include "llvm/Target/TargetMachine.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_compiler.h"

namespace paddle {
namespace piano {
namespace backends {

// NvptxCompiler compile llvm ir to executable binary code for nvidia gpu.

class NvptxCompiler : public LlvmCompiler {
 public:
  NvptxCompiler() = default;
  ~NvptxCompiler() {}

 protected:
  void Optimize(std::unique_ptr<note::Module>&) override;
  void Compile(std::unique_ptr<llvm::Module>&, KernelExecutors&) override;

 private:
  void OptimizeLlvmIR(std::unique_ptr<llvm::Module>&);
  std::unique_ptr<llvm::TargetMachine> GetTargetMachine(llvm::Triple);
  std::string ConverToPtx(std::unique_ptr<llvm::Module>&);
  void GetCuFunction(const std::string&, KernelExecutors&);

  std::string GetLlvmTarget() const { return ""; }
  std::string GetLlvmDataLayout() const { return ""; }

 private:
  static std::once_flag call_once_flag_;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
