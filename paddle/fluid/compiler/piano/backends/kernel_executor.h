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

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace piano {
namespace backends {

// kernel type
enum KernelType {
  BatchNormGradKernel = 0,
  BatchNormInference,
  BatchNormTrainingKernel,
  ConvolutionKernel,
  DotKernel,
  JitKernel,
};

// datatype dev-stream
struct ExecutionContext {};

// KernelExecutor is a kernel executor, it includes kernel information.
class KernelExecutor {
 public:
  KernelExecutor();
  virtual ~KernelExecutor();
  virtual void run(ExecutionContext&) = 0;

 public:
  KernelType GetKernelType() const { return kernel_type_; }
  std::string GetKernelName() const { return kernel_name_; }
  const std::vector<std::string>& GetInputNames() const { return input_names_; }
  const std::vector<std::string>& GetOutputNames() const {
    return output_names_;
  }

  void SetKernelType(KernelType kernel_type) { kernel_type_ = kernel_type; }
  void SetKernelName(const std::string& kernel_name) {
    kernel_name_ = kernel_name;
  }
  void SetInputNames(const std::vector<std::string>& input_name) {
    input_names_ = input_name;
  }
  void SetOutputNames(const std::vector<std::string>& output_name) {
    output_names_ = output_name;
  }

 protected:
  KernelType kernel_type_;
  std::string kernel_name_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
};

using KernelExecutors = std::vector<std::unique_ptr<KernelExecutor>>;

}  // namespace backends
}  // namespace piano
}  // namespace paddle
