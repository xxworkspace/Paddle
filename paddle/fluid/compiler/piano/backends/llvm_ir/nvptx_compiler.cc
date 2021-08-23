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
#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>
#include <unordered_map>
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_executable.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace piano {
namespace backends {

namespace nvptx {

void InitLlvmNvptxContextOnce() {
  LLVMInitializeNVPTXTarget();
  LLVMInitializeNVPTXTargetInfo();
  LLVMInitializeNVPTXTargetMC();
  LLVMInitializeNVPTXAsmPrinter();
}

std::string GetComputeCapability() {
  int device_id = 0;
  PADDLE_ENFORCE_EQ(
      cudaGetDevice(&device_id), 0,
      platform::errors::Unavailable("Cuda device is unavailable!"));
  cudaDeviceProp device_prop;
  PADDLE_ENFORCE_EQ(
      cudaGetDeviceProperties(&device_prop, device_id), 0,
      platform::errors::Fatal("Fail to get cuda device properties!"));
  int major = device_prop.major;
  int minor = device_prop.minor;
  return "sm_" + std::to_string(major * 10 + minor);
}

//
class CuModuleRegistry {
 public:
  ~CuModuleRegistry() {}
  static CuModuleRegistry& GetCuModuleRegistry() {
    static CuModuleRegistry registry_;
    return registry_;
  }

  // registry ptx with note::Module's name
  void RegistryCuModule(const std::string& module_name,
                        const std::string& ptx) {
    PADDLE_ENFORCE_EQ(
        ptxs.count(module_name), 0,
        platform::errors::AlreadyExists(
            "note::Module name = %s has be registered!", module_name));
    ptxs[module_name] = ptx;
  }

  // get a CUfunction from primary context in device_id.
  CUfunction GetCuFunction(const std::string& module_name,
                           const std::string& func_name) {
    PADDLE_ENFORCE_EQ(ptxs.count(module_name), 1,
                      platform::errors::Unavailable(
                          "note::Module name = %s is not found!", module_name));
    // get current device id.
    int device_id = 0;
    PADDLE_ENFORCE_EQ(
        cudaGetDevice(&device_id), 0,
        platform::errors::Unavailable("Cuda device is unavailable!"));

    // module_name + "_" + str(device_id)
    std::string module_device_id =
        module_name + "_" + std::to_string(device_id);
    if (cu_modules.count(module_device_id) == 0) {
      // As CUmodule is not loaded on current primary context, so load CUmodule
      // first.
      std::mutex mtx;
      std::lock_guard<std::mutex> lock(mtx);
      {
        if (cu_modules.count(module_device_id) == 0) {
          CUdevice device;
          CUcontext context;
          // get current CUdevice.
          PADDLE_ENFORCE_EQ(
              cuDeviceGet(&device, device_id), 0,
              platform::errors::Fatal("Fail to get CUdevice on device = %d",
                                      device_id));
          // get current primary CUcontext.
          PADDLE_ENFORCE_EQ(cuCtxGetCurrent(&context), 0,
                            platform::errors::Fatal("Fail to get CUcontext!"));
          // retain primary context for driver api to use.
          PADDLE_ENFORCE_EQ(
              cuDevicePrimaryCtxRetain(&context, device), 0,
              platform::errors::Fatal("Fail to Retain PrimaryCtx!"));
          // load CUmodule from ptx.
          PADDLE_ENFORCE_EQ(
              cuModuleLoadData(&cu_modules[module_device_id],
                               ptxs[module_name].c_str()),
              0, platform::errors::Fatal("note::Module name = %s fail to load "
                                         "CUmodule on device_id = %d !",
                                         module_name, device_id));
        }
      }
    }

    // As CUmodule is loaded, using function name to retrival the CUfuction.
    CUfunction cu_func;
    PADDLE_ENFORCE_EQ(
        cuModuleGetFunction(&cu_func, cu_modules[module_device_id],
                            func_name.c_str()),
        0, platform::errors::Unavailable(
               "note::Module name = %s fail to find CUfunction name = %s",
               module_name, func_name));
    return cu_func;
  }

 private:
  CuModuleRegistry() {}
  DISABLE_COPY_AND_ASSIGN(CuModuleRegistry);
  // CUmodule map for each module and ptx.
  // KEY = module name + "_" + std::to_string(device_id).
  std::unordered_map<std::string, CUmodule> cu_modules;
  // ptx map for each module.
  std::unordered_map<std::string, std::string> ptxs;

  // free CUmodule.
  class Garbage {
   public:
    ~Garbage() noexcept(false) {
      int count = 0;
      PADDLE_ENFORCE_EQ(cudaGetDeviceCount(&count), 0,
                        platform::errors::Fatal("Fail to get device count!"));
      auto& cumodule_registry = GetCuModuleRegistry();
      for (auto& p : cumodule_registry.ptxs) {
        for (int idx = 0; idx < count; ++idx) {
          std::string module_device_id = p.first + "_" + std::to_string(idx);
          PADDLE_ENFORCE_EQ(
              cudaSetDevice(idx), 0,
              platform::errors::Fatal("Fail to set device id = %d", idx));
          if (cumodule_registry.cu_modules.count(module_device_id) > 0) {
            PADDLE_ENFORCE_EQ(
                cuModuleUnload(cumodule_registry.cu_modules[module_device_id]),
                0, platform::errors::Fatal("Fail to unload CUmodule name = %s",
                                           module_device_id));
          }
        }
      }
    }
  } garbage_;
};

}  // namespace nvptx

void NvptxCompiler::Optimize(note::Module*) {}

void NvptxCompiler::Compile(const note::Module&, llvm::Module* llvm_module,
                            KernelExecutableMap* kernel_executable_map) {
  // optimize llvm ir
  OptimizeLlvmIR(llvm_module);
  // convert to ptx
  auto ptx = ConverToPtx(llvm_module);
  // TODO(sunli) : note::Module name
  nvptx::CuModuleRegistry::GetCuModuleRegistry().RegistryCuModule("", ptx);
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
  int optimization_level = 2;
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
  std::call_once(call_once_flag_, nvptx::InitLlvmNvptxContextOnce);
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

}  // namespace backends
}  // namespace piano
}  // namespace paddle
