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
#include <unordered_map>
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

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char* msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
    }                                                                   \
  }

#define CUDA_CALL(func)                                       \
  {                                                           \
    cudaError_t e = (func);                                   \
    ICHECK(e == cudaSuccess || e == cudaErrorCudartUnloading) \
        << "CUDA: " << cudaGetErrorString(e);                 \
  }

void InitLlvmNvptxContext() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

std::string GetComputeCapability() {
  int device_id = 0;
  CUDA_CALL(cudaGetDevice(&device_id));
  cudaDeviceProp device_prop;
  CUDA_CALL(cudaGetDeviceProperties(&device_prop, device_id));
  int major = device_prop.major;
  int minor = device_prop.minor;
  return std::to_string(major * 10 + minor);
}

class CuModuleRegistry {
 public:
  static CuModuleRegistry GetCuModuleRegistry() {
    static CuModuleRegistry registry_;
    return registry_;
  }

  void RegistryCuModule(const std::int64_t module_global_id,
                        const std::string& ptx) {
    ptxs[module_global_id] = ptx;
  }

  CUFunction GetCuFunction(const std::int64_t module_global_id,
                           const std::string& func_name) {
    ASSERT_NE(ptxs.count(module_global_id), 0);
    if (cu_modules.count(module_global_id) == 0) {
      auto& ptx = ptxs[module_global_id];
      CUDA_DRIVER_CALL(
          cuModuleLoadData(&cu_modules[module_global_id], ptx.c_str()));
    }

    CUFunction cu_func;
    CUDA_DRIVER_CALL(cuModuleGetFunction(&cu_func, cu_modules[module_global_id],
                                         func_name.c_str()));
    return cu_func;
  }

 private:
  CuModuleRegistry();
  ~CuModuleRegistry();
  std::unordered_map<std::int64_t, CUmodule> cu_modules;
  std::unordered_map<std::int64_t, std::string> ptxs;

  class Garbage {
    ~Garbage {
      auto registry_ = GetCuModuleRegistry();
      for (auto& cu_module : cu_modules) {
        CUDA_DRIVER_CALL(cuModuleUnload(cu_module.second));
      }
    }
  } garbage_;
};

}  // namespace nvptx

void NvptxCompiler::Optimize(note::Module*) {}

void NvptxCompiler::Compile(llvm::Module* llvm_module,
                            KernelExecutableMap* kernel_executable_map) {
  // optimize llvm ir
  OptimizeLlvmIR(llvm_module);
  // convert to ptx
  auto ptx = ConverToPtx(llvm_module);
  // get cu function
  // GetCuFunction(ptx, kernel_executable_map);
}

void NvptxCompiler::OptimizeLlvmIR(llvm::Module* llvm_module) {}

std::unique_ptr<llvm::TargetMachine> NvptxCompiler::GetTargetMachine(
    llvm::Triple llvm_triple) {
  std::string error = "";
  const llvm::Target* target =
      llvm::TargetRegistry::lookupTarget("", llvm_triple, error);

  llvm::TargetOptions target_options =
      llvm::codegen::InitTargetOptionsFromCodeGenFlags(llvm::Triple());

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

std::string NvptxCompiler::ConverToPtx(llvm::Module* llvm_module) {
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
  pass_manager.run(*llvm_module);
  return ptx;
}

void NvptxCompiler::RegistryCuModule(const int64_t global_id,
                                     const std::string& ptx) {
  nvtpx::CuModuleRegistry::GetCuModuleRegistry().RegistryCuModule(global_id,
                                                                  ptx);
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
