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

namespace note {
class Instruction;
}

class PrimitiveIrEmitter : public NoteVisitorBase<const note::Instruction*> {
 public:
  PrimitiveIrEmitter() {}
  virtual ~PrimitiveIrEmitter() {}

  virtual void VisitElementwiseUnary(const note::Instruction*) = 0;
  virtual void VisitElementwiseBinary(const note::Instruction*) = 0;

  // Scalar op
  virtual void VisitConstant(const note::Instruction*) = 0;

  // ops can be replaced by library
  void VisitBatchNormGrad(const note::Instruction*) override{};
  void VisitBatchNormInference(const note::Instruction*) override{};
  void VisitBatchNormTraining(const note::Instruction*) override{};
  void VisitConvolution(const note::Instruction*) override{};
  void VisitDot(const note::Instruction*) override{};

  // Unary
  virtual void VisitBroadcast(const note::Instruction*) = 0;
  virtual void VisitCopy(const note::Instruction*) = 0;
  virtual void VisitReshape(const note::Instruction*) = 0;
  virtual void VisitReverse(const note::Instruction*) = 0;
  virtual void VisitSlice(const note::Instruction*) = 0;
  virtual void VisitTranspose(const note::Instruction*) = 0;

  // Unary Compute
  void VisitCast(const note::Instruction*) override;
  void VisitExp(const note::Instruction*) override;
  void VisitLog(const note::Instruction*) override;
  void VisitNegative(const note::Instruction*) override;
  void VisitNot(const note::Instruction*) override;
  void VisitRsqrt(const note::Instruction*) override;
  void VisitSqrt(const note::Instruction*) override;

  // Binary
  void VisitAdd(const note::Instruction*) override;
  void VisitAnd(const note::Instruction*) override;
  void VisitCompare(const note::Instruction*) override;
  void VisitDivide(const note::Instruction*) override;
  void VisitMaximum(const note::Instruction*) override;
  void VisitMinimum(const note::Instruction*) override;
  void VisitMultiply(const note::Instruction*) override;
  void VisitOr(const note::Instruction*) override;
  void VisitSubtract(const note::Instruction*) override;
  void VisitXor(const note::Instruction*) override;

  // other
  virtual void VisitSelect(const note::Instruction*) = 0;
  virtual void VisitConcatenate(const note::Instruction*) = 0;
  virtual void VisitReduce(const note::Instruction*) = 0;
  virtual void VisitRng(const note::Instruction*) = 0;
  virtual void VisitSort(const note::Instruction*) = 0;
  virtual void VisitTuple(const note::Instruction*) = 0;

  std::vector<PrimitiveIrGenerator> GetPrimitiveIrGenerators();

 protected:
  std::vector<PrimitiveIrGenerator> primitive_ir_generators_;
};

}  // namespace piano
}  // namespace paddle
