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
#include <mutex>
#include <unordered_map>
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/executable_pool.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_executable.h"

namespace paddle {
namespace piano {
namespace backends {

namespace nvptx {

void InitLlvmNvptxContext() {
  // init target
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
  // get pass registry
  llvm::PassRegistry* pass_registry = llvm::PassRegistry::getPassRegistry();
  // registry optimize pass
  llvm::initializeCore(*pass_registry);
  llvm::initializeCodeGen(*pass_registry);
  llvm::initializeScalarOpts(*pass_registry);
  llvm::initializeObjCARCOpts(*pass_registry);
  llvm::initializeVectorization(*pass_registry);
  llvm::initializeIPO(*pass_registry);
  llvm::initializeAnalysis(*pass_registry);
  llvm::initializeTransformUtils(*pass_registry);
  llvm::initializeInstCombine(*pass_registry);
  llvm::initializeInstrumentation(*pass_registry);
  llvm::initializeTarget(*pass_registry);
  llvm::initializeCodeGenPreparePass(*pass_registry);
}

std::string GetComputeCapability() {
  int device_id = platform::GetCurrentDeviceId();
  int capability = platform::GetCUDAComputeCapability(device_id);
  return "sm_" + std::to_string(capability);
}

}  // namespace nvptx

void NvptxCompiler::Optimize(note::Module*) {
  // TODO(sunli) : optimize pass
}

void NvptxCompiler::Compile(const note::Module& note_module,
                            llvm::Module* llvm_module) {
  // set triple and datalayout
  llvm_module->setTargetTriple(llvm::StringRef(GetLlvmTriple()));
  llvm_module->setDataLayout(llvm::StringRef(GetLlvmDataLayout()));

  // convert to ptx
  auto ptx = CompileToPtx(llvm_module);

  // registry ptx
  CumodulePool::Instance().Insert(note_module.name(), ptx);
}

std::unique_ptr<llvm::TargetMachine> NvptxCompiler::GetTargetMachine(
    llvm::Triple llvm_triple) {
  // lookupTarget
  std::string error = "";
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", llvm_triple, error);

  // target_options
  llvm::TargetOptions target_options;
  target_options.AllowFPOpFusion = llvm::FPOpFusion::Fast;
  target_options.UnsafeFPMath = false;
  target_options.NoInfsFPMath = false;
  target_options.NoNaNsFPMath = true;
  target_options.FloatABIType = llvm::FloatABI::Soft;
  target_options.MCOptions.AsmVerbose = false;

  std::string compute_capability = nvptx::GetComputeCapability();

  // codegen opt level : llvm::CodeGenOpt::Less/Default/Aggressive/None
  llvm::CodeGenOpt::Level codegen_opt_level = llvm::CodeGenOpt::Default;

  // triple mcpu mattr opt
  return std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
      llvm_triple.str(), llvm::StringRef(compute_capability),
      llvm::StringRef("+ptx60"), target_options, llvm::Reloc::PIC_,
      llvm::CodeModel::Medium, codegen_opt_level));
}

std::string NvptxCompiler::CompileToPtx(llvm::Module* llvm_module) {
  // init context
  static std::once_flag call_once_flag_;
  std::call_once(call_once_flag_, nvptx::InitLlvmNvptxContext);

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
  // TODO(sunli) : coredump catch!
  pass_manager.run(*llvm_module);

  return ptx;
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
