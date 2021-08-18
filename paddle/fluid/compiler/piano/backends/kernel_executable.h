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

#include <memory>
#include <string>
#include <unordered_map>
#include "paddle/fluid/compiler/piano/note/instruction.h"

namespace paddle {
namespace piano {
namespace backends {

// kernel type.
enum class KernelType {
  BatchNormGradKernel = 0,
  BatchNormInference,
  BatchNormTrainingKernel,
  ConvolutionKernel,
  DotKernel,
  JitKernel,
};

// executable context for KernelExecutable.
struct ExecutableContext {};

// KernelExecutable is a kernel execution class, it includes kernel information.
// Each 'KernelType' need define a derived class which inherit
// 'KernelExecutable'
// and overwrite the virtual function 'Run'.
class KernelExecutable {
 public:
  KernelExecutable(const note::Instruction&) {}
  virtual ~KernelExecutable();
  virtual void Run(const ExecutableContext&) = 0;

 public:
  void Reset(const note::Instruction&) {}

  KernelType GetKernelType() const { return kernel_type_; }
  std::string GetKernelName() const { return kernel_name_; }
  const std::vector<std::string>& GetInputNames() const { return input_names_; }
  const std::vector<std::string>& GetOutputNames() const {
    return output_names_;
  }

 protected:
  int64_t global_id_;
  KernelType kernel_type_;
  std::string kernel_name_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

using KernelExecutableMap =
    std::unordered_map<int64_t, std::unique_ptr<KernelExecutable>>;

}  // namespace backends
}  // namespace piano
}  // namespace paddle
