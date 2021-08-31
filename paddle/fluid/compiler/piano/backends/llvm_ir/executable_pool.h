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

#include <atomic>
#include "paddle/fluid/compiler/piano/backends/kernel_executable.h"
#include "paddle/fluid/platform/dynload/cuda_driver.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace piano {
namespace backends {

// To be removed in future commit by using PADDLE_ENFORCE_CUDA_SUCCESS.
#define CHECK_CUDA_DRIVER_SUCCESS(curesult)                             \
  if (curesult != CUDA_SUCCESS) {                                       \
    const char* msg;                                                    \
    platform::dynload::cuGetErrorString(curesult, &msg);                \
    PADDLE_THROW(platform::errors::External("cu driver error(%d) : %s", \
                                            curesult, msg));            \
  }

class CumodulePool {
 public:
  ~CumodulePool() {
    int count = platform::GetCUDADeviceCount();
    auto& cumodule_pool = Instance();
    for (auto& p : cumodule_pool.ptx_map_) {
      for (int idx = 0; idx < count; ++idx) {
        std::string module_device_id = p.first + "_" + std::to_string(idx);
        if (cumodule_pool.cumodule_map_.count(module_device_id) > 0) {
          try {
            platform::dynload::cuModuleUnload(
                cumodule_pool.cumodule_map_[module_device_id]);
          } catch (...) {
            platform::errors::External(
                "cu driver error : cuModuleUnload fail when device id = %d",
                idx);
          }
        }
      }
    }
  }

  static CumodulePool& Instance() {
    static CumodulePool cumodule_pool;
    return cumodule_pool;
  }

  // registry ptx with note::Module's name
  void Insert(const std::string& module_name, const std::string& ptx) {
    PADDLE_ENFORCE_EQ(
        ptx_map_.count(module_name), 0,
        platform::errors::AlreadyExists(
            "note::Module name = %s has be inserted!", module_name));

    std::lock_guard<std::mutex> lock(mutex_);
    ptx_map_[module_name] = ptx;
  }

  // get a CUfunction from primary context in device_id.
  CUfunction GetCuFunction(const std::string& module_name,
                           const std::string& func_name) {
    PADDLE_ENFORCE_EQ(ptx_map_.count(module_name), 1,
                      platform::errors::Unavailable(
                          "note::Module name = %s is not found!", module_name));
    // get current device id.
    int device_id = platform::GetCurrentDeviceId();
    std::string module_device_id =
        module_name + "_" + std::to_string(device_id);
    if (cumodule_map_.count(module_device_id) == 0) {
      // As CUmodule is not loaded on current primary context, so load CUmodule
      // first.
      std::lock_guard<std::mutex> lock(mutex_);
      {
        if (cumodule_map_.count(module_device_id) == 0) {
          CUdevice device;
          CUcontext context;
          // get current CUdevice.
          CHECK_CUDA_DRIVER_SUCCESS(
              platform::dynload::cuDeviceGet(&device, device_id));
          // get current primary CUcontext.
          CHECK_CUDA_DRIVER_SUCCESS(
              platform::dynload::cuCtxGetCurrent(&context));
          // retain primary context for driver api to use.
          CHECK_CUDA_DRIVER_SUCCESS(
              platform::dynload::cuDevicePrimaryCtxRetain(&context, device));
          // load CUmodule from ptx
          std::cerr << ptx_map_[module_name] << std::endl;
          CHECK_CUDA_DRIVER_SUCCESS(platform::dynload::cuModuleLoadData(
              &cumodule_map_[module_device_id], ptx_map_[module_name].c_str()));
        }
      }
    }

    // As CUmodule is loaded, using function name to retrival the CUfuction.
    CUfunction cu_func;
    CHECK_CUDA_DRIVER_SUCCESS(platform::dynload::cuModuleGetFunction(
        &cu_func, cumodule_map_[module_device_id], func_name.c_str()));
    return cu_func;
  }

 private:
  CumodulePool() = default;

  // CUmodule map for each module and ptx.
  std::mutex mutex_;
  // KEY = module name + "_" + std::to_string(device_id).
  std::unordered_map<std::string, CUmodule> cumodule_map_;
  // ptx map for each module.
  std::unordered_map<std::string, std::string> ptx_map_;

  DISABLE_COPY_AND_ASSIGN(CumodulePool);
};

class ExecutablePool {
 public:
  static ExecutablePool& Instance() {
    static ExecutablePool executable_pool;
    return executable_pool;
  }

  void Insert(KernelExecutable* kernel_executable) {
    PADDLE_ENFORCE_NOT_NULL(kernel_executable,
                            platform::errors::PreconditionNotMet(
                                "KernelExecutable pointer can't be nullptr!"));
    static std::mutex mtx;
    std::lock_guard<std::mutex> lock(mtx);
    kernel_executables_.push_back(kernel_executable);
  }

  ~ExecutablePool() {
    for (auto it : kernel_executables_) {
      delete it;
    }
  }

 private:
  ExecutablePool() = default;
  std::vector<KernelExecutable*> kernel_executables_;
  DISABLE_COPY_AND_ASSIGN(ExecutablePool);
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
