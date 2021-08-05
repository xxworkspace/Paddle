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

class NoteInstruction;

class PrimitiveIrEmitter : public NoteVisitorBase<const NoteInstruction*> {
 public:
  PrimitiveIrEmitter() {}
  virtual ~PrimitiveIrEmitter() {}

  virtual void VisitElementwiseUnary(const NoteInstruction*) = 0;
  virtual void VisitElementwiseBinary(const NoteInstruction*) = 0;

  // Scalar op
  virtual void VisitConstant(const NoteInstruction*) = 0;

  // ops can be replaced by library
  void VisitBatchNormGrad(const NoteInstruction*) override{};
  void VisitBatchNormInference(const NoteInstruction*) override{};
  void VisitBatchNormTraining(const NoteInstruction*) override{};
  void VisitConvolution(const NoteInstruction*) override{};
  void VisitDot(const NoteInstruction*) override{};

  // Unary
  virtual void VisitBroadcast(const NoteInstruction*) = 0;
  virtual void VisitCopy(const NoteInstruction*) = 0;
  virtual void VisitReshape(const NoteInstruction*) = 0;
  virtual void VisitReverse(const NoteInstruction*) = 0;
  virtual void VisitSlice(const NoteInstruction*) = 0;
  virtual void VisitTranspose(const NoteInstruction*) = 0;

  // Unary Compute
  void VisitCast(const NoteInstruction*) override;
  void VisitExp(const NoteInstruction*) override;
  void VisitLog(const NoteInstruction*) override;
  void VisitNegative(const NoteInstruction*) override;
  void VisitNot(const NoteInstruction*) override;
  void VisitRsqrt(const NoteInstruction*) override;
  void VisitSqrt(const NoteInstruction*) override;

  // Binary
  void VisitAdd(const NoteInstruction*) override;
  void VisitAnd(const NoteInstruction*) override;
  void VisitCompare(const NoteInstruction*) override;
  void VisitDivide(const NoteInstruction*) override;
  void VisitMaximum(const NoteInstruction*) override;
  void VisitMinimum(const NoteInstruction*) override;
  void VisitMultiply(const NoteInstruction*) override;
  void VisitOr(const NoteInstruction*) override;
  void VisitSubtract(const NoteInstruction*) override;
  void VisitXor(const NoteInstruction*) override;

  // other
  virtual void VisitSelect(const NoteInstruction*) = 0;
  virtual void VisitConcatenate(const NoteInstruction*) = 0;
  virtual void VisitReduce(const NoteInstruction*) = 0;
  virtual void VisitRng(const NoteInstruction*) = 0;
  virtual void VisitSort(const NoteInstruction*) = 0;
  virtual void VisitTuple(const NoteInstruction*) = 0;

  std::vector<PrimitiveIrGenerator> GetPrimitiveIrGenerators();

 protected:
  std::vector<PrimitiveIrGenerator> primitive_ir_generators_;
};

}  // namespace piano
}  // namespace paddle
