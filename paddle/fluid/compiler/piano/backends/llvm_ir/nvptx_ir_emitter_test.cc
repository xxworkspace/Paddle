/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_ir_emitter.h"
#include "glog/logging.h"
#include "gtest/gtest.h"

namespace paddle {
namespace piano {
namespace backends {

TEST(NvptxIrEmitter, TestOp) {
  // llvm module
  // llvm::LLVMContext llvm_context;
  // llvm::Module llvm_module("", llvm_context);
  // scheduls
  // Schedules schedules;

  // create ir emitter
  NvptxIrEmitter nvptx_ir_emitter(nullptr, nullptr);

  // Scalar Op
  nvptx_ir_emitter.VisitConstant(nullptr);

  // ops can be replaced by library
  nvptx_ir_emitter.VisitBatchNormGrad(nullptr);
  nvptx_ir_emitter.VisitBatchNormInference(nullptr);
  nvptx_ir_emitter.VisitBatchNormTraining(nullptr);
  nvptx_ir_emitter.VisitConvolution(nullptr);
  nvptx_ir_emitter.VisitDot(nullptr);

  // Unary
  nvptx_ir_emitter.VisitBroadcast(nullptr);
  nvptx_ir_emitter.VisitCast(nullptr);
  nvptx_ir_emitter.VisitCopy(nullptr);
  nvptx_ir_emitter.VisitExp(nullptr);
  nvptx_ir_emitter.VisitLog(nullptr);
  nvptx_ir_emitter.VisitNegative(nullptr);
  nvptx_ir_emitter.VisitNot(nullptr);
  nvptx_ir_emitter.VisitReshape(nullptr);
  nvptx_ir_emitter.VisitReverse(nullptr);
  nvptx_ir_emitter.VisitRsqrt(nullptr);
  nvptx_ir_emitter.VisitSlice(nullptr);
  nvptx_ir_emitter.VisitSqrt(nullptr);
  nvptx_ir_emitter.VisitTranspose(nullptr);

  // Binary
  nvptx_ir_emitter.VisitAdd(nullptr);
  nvptx_ir_emitter.VisitAnd(nullptr);
  nvptx_ir_emitter.VisitCompare(nullptr);
  nvptx_ir_emitter.VisitDivide(nullptr);
  nvptx_ir_emitter.VisitMaximum(nullptr);
  nvptx_ir_emitter.VisitMinimum(nullptr);
  nvptx_ir_emitter.VisitMultiply(nullptr);
  nvptx_ir_emitter.VisitOr(nullptr);
  nvptx_ir_emitter.VisitSubtract(nullptr);
  nvptx_ir_emitter.VisitXor(nullptr);

  // Others
  nvptx_ir_emitter.VisitSelect(nullptr);
  nvptx_ir_emitter.VisitConcatenate(nullptr);
  nvptx_ir_emitter.VisitReduce(nullptr);
  nvptx_ir_emitter.VisitRng(nullptr);
  nvptx_ir_emitter.VisitSort(nullptr);
  nvptx_ir_emitter.VisitTuple(nullptr);
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
