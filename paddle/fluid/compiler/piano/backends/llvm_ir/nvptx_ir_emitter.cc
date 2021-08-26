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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_ir_emitter.h"
#include "paddle/fluid/platform/errors.h"

namespace paddle {
namespace piano {
namespace backends {

void NvptxIrEmitter::VisitBatchNormGrad(const note::Instruction& instr) {
  PADDLE_THROW(
      platform::errors::Unimplemented("BatchNormGrad is unimplemented!"));
}

void NvptxIrEmitter::VisitBatchNormInference(const note::Instruction& instr) {
  PADDLE_THROW(
      platform::errors::Unimplemented("BatchNormInference is unimplemented!"));
}

void NvptxIrEmitter::VisitBatchNormTraining(const note::Instruction& instr) {
  PADDLE_THROW(
      platform::errors::Unimplemented("BatchNormTraining is unimplemented!"));
}

void NvptxIrEmitter::VisitConvolution(const note::Instruction& instr) {
  PADDLE_THROW(
      platform::errors::Unimplemented("Convolution is unimplemented!"));
}

void NvptxIrEmitter::VisitDot(const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("VisitDot is unimplemented!"));
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle