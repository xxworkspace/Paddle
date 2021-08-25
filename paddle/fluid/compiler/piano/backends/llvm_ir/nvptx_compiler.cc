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
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_executable.h"
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/gpu_info.h"

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

class CUmodulePool {
 public:
  ~CUmodulePool() {}
  static CUmodulePool& GetCUmodulePool() {
    static CUmodulePool cumodule_pool;
    return cumodule_pool;
  }

  // registry ptx with note::Module's name
  void InsertCUmodule(const std::string& module_name, const std::string& ptx) {
    PADDLE_ENFORCE_EQ(
        ptx_map_.count(module_name), 0,
        platform::errors::AlreadyExists(
            "note::Module name = %s has be inserted!", module_name));

    std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    ptx_map_[module_name] = ptx;
  }

  // get a CUfunction from primary context in device_id.
  CUfunction GetCUfunction(const std::string& module_name,
                           const std::string& func_name) {
    PADDLE_ENFORCE_EQ(ptx_map_.count(module_name), 1,
                      platform::errors::Unavailable(
                          "note::Module name = %s is not found!", module_name));
    // get current device id.
    int device_id = platform::GetCurrentDeviceId();
    // module_name + "_" + str(device_id)
    std::string module_device_id =
        module_name + "_" + std::to_string(device_id);
    if (cumodule_map_.count(module_device_id) == 0) {
      // As CUmodule is not loaded on current primary context, so load CUmodule
      // first.
      std::mutex mtx;
      std::lock_guard<std::mutex> lock(mtx);
      {
        if (cumodule_map_.count(module_device_id) == 0) {
          CUdevice device;
          CUcontext context;
          // get current CUdevice.
          PADDLE_ENFORCE_EQ(
              platform::dynload::cuDeviceGet(&device, device_id), CUDA_SUCCESS,
              platform::errors::External(
                  "Fail to get CUdevice width device id = %d !", device_id));
          // get current primary CUcontext.
          PADDLE_ENFORCE_EQ(
              platform::dynload::cuCtxGetCurrent(&context), CUDA_SUCCESS,
              platform::errors::External("Fail to get CUcontext!"));
          // retain primary context for driver api to use.
          PADDLE_ENFORCE_EQ(
              platform::dynload::cuDevicePrimaryCtxRetain(&context, device),
              CUDA_SUCCESS,
              platform::errors::External("Fail to retain CUcontext!"));
          // load CUmodule from ptx.
          PADDLE_ENFORCE_EQ(platform::dynload::cuModuleLoadData(
                                &cumodule_map_[module_device_id],
                                ptx_map_[module_name].c_str()),
                            CUDA_SUCCESS, platform::errors::External(
                                              "Fail to load CUmodule!"));
        }
      }
    }

    // As CUmodule is loaded, using function name to retrival the CUfuction.
    CUfunction cu_func;
    PADDLE_ENFORCE_EQ(
        platform::dynload::cuModuleGetFunction(
            &cu_func, cumodule_map_[module_device_id], func_name.c_str()),
        CUDA_SUCCESS,
        platform::errors::External("Fail to get CUfunction = %s !", func_name));
    return cu_func;
  }

 private:
  CUmodulePool() {}
  DISABLE_COPY_AND_ASSIGN(CUmodulePool);
  // CUmodule map for each module and ptx.
  // KEY = module name + "_" + std::to_string(device_id).
  std::unordered_map<std::string, CUmodule> cumodule_map_;
  // ptx map for each module.
  std::unordered_map<std::string, std::string> ptx_map_;

  // free CUmodule.
  class Garbage {
   public:
    ~Garbage() noexcept(false) {
      int count = platform::GetCUDADeviceCount();
      auto& cumodule_registry = GetCUmodulePool();
      for (auto& p : cumodule_registry.ptx_map_) {
        for (int idx = 0; idx < count; ++idx) {
          platform::SetDeviceId(idx);
          std::string module_device_id = p.first + "_" + std::to_string(idx);
          if (cumodule_registry.cumodule_map_.count(module_device_id) > 0) {
            PADDLE_ENFORCE_EQ(
                platform::dynload::cuModuleUnload(
                    cumodule_registry.cumodule_map_[module_device_id]),
                CUDA_SUCCESS,
                platform::errors::External("Fail to unload CUmodule!"));
          }
        }
      }
    }
  } garbage_;
};

// Call first time
static CUmodulePool& cumodule_pool_init = CUmodulePool::GetCUmodulePool();
}  // namespace nvptx

void NvptxCompiler::Optimize(note::Module*) {}

void NvptxCompiler::Compile(const note::Module& note_module,
                            llvm::Module* llvm_module,
                            KernelExecutableMap* kernel_executable_map) {
  // set triple and datalayout
  llvm_module->setTargetTriple(llvm::StringRef(GetLlvmTriple()));
  llvm_module->setDataLayout(llvm::StringRef(GetLlvmDataLayout()));

  // convert to ptx
  auto ptx = CompileToPtx(llvm_module);

  // registry ptx
  nvptx::CUmodulePool::GetCUmodulePool().InsertCUmodule(note_module.name(),
                                                        ptx);
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
  std::cerr << compute_capability << std::endl;

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
  pass_manager.run(*llvm_module);
  return ptx;
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
