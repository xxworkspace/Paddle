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
enum class KernelType : std::uint32_t {
  kBatchNormGradKernel = 0,
  kBatchNormInferenceKernel,
  kBatchNormTrainingKernel,
  kConvolutionKernel,
  kDotKernel,
  kJitKernel,
};

// KernelExecutable is a kernel execution class, it includes kernel information.
// Each KernelType need define a derived class which inherit KernelExecutable
// and overwrite the virtual function Run.
// By call function Run in KernelExecutor to execute the binary code.
class KernelExecutable {
 public:
  explicit KernelExecutable(const note::Instruction& note_instruction) {
    Reset(note_instruction);
  }
  virtual ~KernelExecutable() {}
  // launch kernel
  // TODO(sunli) : handle different data type flaot/half/...
  virtual void Launch(std::vector<void*>&, void*) = 0;

 public:
  void Reset(const note::Instruction& note_instruction) {
    kernel_name_ = note::GetOpName(note_instruction.opcode()) + "_" +
                   std::to_string(note_instruction.global_id());
    // initialize KernelType
    switch (note_instruction.opcode()) {
      case note::OpCode::kBatchNormGrad:
        kernel_type_ = KernelType::kBatchNormGradKernel;
        break;
      case note::OpCode::kBatchNormInference:
        kernel_type_ = KernelType::kBatchNormInferenceKernel;
        break;
      case note::OpCode::kBatchNormTraining:
        kernel_type_ = KernelType::kBatchNormTrainingKernel;
        break;
      case note::OpCode::kConvolution:
        kernel_type_ = KernelType::kConvolutionKernel;
        break;
      case note::OpCode::kDot:
        kernel_type_ = KernelType::kDotKernel;
        break;
      default:
        kernel_type_ = KernelType::kJitKernel;
        break;
    }

    // get op input global_id and name
    for (auto operand : note_instruction.operands()) {
      input_names_.push_back(operand->name());
    }

    // TODO(sunli) : for multi output!!!
    output_names_.push_back(note_instruction.name());
  }

  std::string GetKernelName() const { return kernel_name_; }
  KernelType GetKernelType() const { return kernel_type_; }
  const std::vector<std::string>& GetInputNames() const { return input_names_; }
  const std::vector<std::string>& GetOutputNames() const {
    return output_names_;
  }

 protected:
  // instruction name
  std::string kernel_name_;
  // executable type
  KernelType kernel_type_;
  // input order
  std::vector<std::string> input_names_;
  // output order
  std::vector<std::string> output_names_;
};

// KernelExecutableMap is a map of KernelExecutable.
using KernelExecutableMap = std::unordered_map<std::string, KernelExecutable*>;

}  // namespace backends
}  // namespace piano
}  // namespace paddle
