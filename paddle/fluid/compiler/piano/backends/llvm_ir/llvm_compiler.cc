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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_compiler.h"
#include "llvm/IR/Verifier.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

KernelExecutableMap LlvmCompiler::Apply(note::Module* note_module) {
  // using pass optimize the note module
  Optimize(note_module);

  // create llvm module
  llvm::LLVMContext context;
  llvm::Module llvm_module(note_module->name(), context);

  // create kernel executor
  KernelExecutableMap kernel_executable_map;

  // conver operator to llvm ir
  ConvertToIr(*note_module, &llvm_module, &kernel_executable_map);

  // verify llvm module
  std::string errors;
  llvm::raw_string_ostream llvm_errors(errors);
  PADDLE_ENFORCE_NE(llvm::verifyModule(llvm_module, &llvm_errors), true,
                    llvm_errors.str());

  // compiler llvm ir to lowring ir
  Compile(*note_module, &llvm_module);

  return kernel_executable_map;
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
