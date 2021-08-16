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
#include <cuda_runtime.h>
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace paddle {
namespace piano {
namespace backends {

namespace nvptx {

void InitLlvmNvptxContext() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

std::string GetComputeCapability() {
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, 0);
  int major = device_prop.major;
  int minor = device_prop.minor;
  return std::to_string(major * 100 + minor);
}

}  // namespace nvptx

void NvptxCompiler::Optimize(std::unique_ptr<note::Module>& note_module) {}

void NvptxCompiler::Compile(std::unique_ptr<llvm::Module>& llvm_module,
                            KernelExecutors& kernel_executors_) {
  // optimize llvm ir
  OptimizeLlvmIR(llvm_module);
  // convert to ptx
  auto ptx = ConverToPtx(llvm_module);
  // get cu function
  GetCuFunction(ptx, kernel_executors_);
}

void NvptxCompiler::OptimizeLlvmIR(std::unique_ptr<llvm::Module>& llvm_module) {
}

std::unique_ptr<llvm::TargetMachine> NvptxCompiler::GetTargetMachine(
    llvm::Triple llvm_triple) {
  std::string error = "";
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", llvm_triple, error);

  llvm::TargetOptions target_options =
      llvm::codegen::InitTargetOptionsFromCodeGenFlags(llvm_triple);

  target_options.MCOptions.AsmVerbose = false;
  std::string compute_capability = nvptx::GetComputeCapability();

  llvm::CodeGenOpt::Level codegen_opt_level;
  int optimization_level = 1;
  switch (optimization_level) {
    case 1:
      codegen_opt_level = llvm::CodeGenOpt::Less;
      break;
    case 2:
      codegen_opt_level = llvm::CodeGenOpt::Default;
      break;
    case 3:
      codegen_opt_level = llvm::CodeGenOpt::Aggressive;
      break;
    default:
      codegen_opt_level = llvm::CodeGenOpt::None;
  }

  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      llvm_triple.str(), llvm::StringRef(compute_capability),
      llvm::StringRef("+ptx60"), target_options,
      llvm::codegen::getExplicitRelocModel(),
      llvm::codegen::getExplicitCodeModel(), codegen_opt_level));
}

std::string NvptxCompiler::ConverToPtx(
    std::unique_ptr<llvm::Module>& llvm_module) {
  // init context
  std::call_once(NvptxCompiler::call_once_flag_, nvptx::InitLlvmNvptxContext);
  // get target machine
  auto target_machine =
      GetTargetMachine(llvm::Triple(llvm_module->getTargetTriple()));
  // create pass
  llvm::legacy::PassManager pass_manager;
  pass_manager.add(new llvm::TargetLibraryInfoWrapperPass(
      llvm::Triple(llvm_module->getTargetTriple())));
  // ptx
  std::string ptx = "";
  llvm::raw_string_ostream string_stream(ptx);
  llvm::buffer_ostream out_stream(string_stream);
  target_machine->addPassesToEmitFile(pass_manager, out_stream, nullptr,
                                      llvm::CGFT_AssemblyFile);
  // run pass
  pass_manager.run(*llvm_module.get());

  return ptx;
}

void NvptxCompiler::GetCuFunction(const std::string& ptx,
                                  KernelExecutors& kernel_executors_) {}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
