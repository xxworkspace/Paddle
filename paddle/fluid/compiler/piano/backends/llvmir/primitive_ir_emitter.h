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

#include <vector>
#include "llvm/IR/Module.h"
#include "paddle/fluid/compiler/piano/backends/llvmir/primitive_ir_generator.h"
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"

namespace paddle {
namespace piano {

class Instruction;

class PrimitiveIrEmitter : public NoteVisitorBase<const Instruction*> {
 public:
  PrimitiveIrEmitter() {}
  virtual ~PrimitiveIrEmitter() {}

  virtual void VisitElementwiseUnary(const Instruction*) = 0;
  virtual void VisitElementwiseBinary(const Instruction*) = 0;

  // Scalar op
  virtual void VisitConstant(const Instruction*) = 0;

  // ops can be replaced by library
  void VisitBatchNormGrad(const Instruction*) override{};
  void VisitBatchNormInference(const Instruction*) override{};
  void VisitBatchNormTraining(const Instruction*) override{};
  void VisitConvolution(const Instruction*) override{};
  void VisitDot(const Instruction*) override{};

  // Unary
  virtual void VisitBroadcast(const Instruction*) = 0;
  virtual void VisitCopy(const Instruction*) = 0;
  virtual void VisitReshape(const Instruction*) = 0;
  virtual void VisitReverse(const Instruction*) = 0;
  virtual void VisitSlice(const Instruction*) = 0;
  virtual void VisitTranspose(const Instruction*) = 0;

  // Unary Compute
  void VisitCast(const Instruction*) override;
  void VisitExp(const Instruction*) override;
  void VisitLog(const Instruction*) override;
  void VisitNegative(const Instruction*) override;
  void VisitNot(const Instruction*) override;
  void VisitRsqrt(const Instruction*) override;
  void VisitSqrt(const Instruction*) override;

  // Binary
  void VisitAdd(const Instruction*) override;
  void VisitAnd(const Instruction*) override;
  void VisitCompare(const Instruction*) override;
  void VisitDivide(const Instruction*) override;
  void VisitMaximum(const Instruction*) override;
  void VisitMinimum(const Instruction*) override;
  void VisitMultiply(const Instruction*) override;
  void VisitOr(const Instruction*) override;
  void VisitSubtract(const Instruction*) override;
  void VisitXor(const Instruction*) override;

  // other
  virtual void VisitSelect(const Instruction*) = 0;
  virtual void VisitConcatenate(const Instruction*) = 0;
  virtual void VisitReduce(const Instruction*) = 0;
  virtual void VisitRng(const Instruction*) = 0;
  virtual void VisitSort(const Instruction*) = 0;
  virtual void VisitTuple(const Instruction*) = 0;

  std::vector<PrimitiveIrGenerator> GetPrimitiveIrGenerators();

 protected:
  std::vector<PrimitiveIrGenerator> primitive_ir_generators_;
};

}  // namespace piano
}  // namespace paddle
