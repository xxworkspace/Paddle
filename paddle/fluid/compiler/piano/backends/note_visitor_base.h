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

namespace paddle {
namespace piano {
namespace backends {

template <typename InstructionType>
class NoteVisitorBase {
 public:
  virtual ~NoteVisitorBase() {}

  // Scalar op
  virtual void VisitConstant(InstructionType) = 0;
  // TODO(sunli): use the pure virtual function instead
  virtual void VisitParameter(InstructionType) = 0;

  // ops can be replaced by library
  virtual void VisitBatchNormGrad(InstructionType) = 0;
  virtual void VisitBatchNormInference(InstructionType) = 0;
  virtual void VisitBatchNormTraining(InstructionType) = 0;
  virtual void VisitConvolution(InstructionType) = 0;
  virtual void VisitDot(InstructionType) = 0;

  // Unary
  virtual void VisitBroadcast(InstructionType) = 0;
  virtual void VisitCast(InstructionType) = 0;
  virtual void VisitCopy(InstructionType) = 0;
  virtual void VisitExp(InstructionType) = 0;
  virtual void VisitLog(InstructionType) = 0;
  virtual void VisitNegative(InstructionType) = 0;
  virtual void VisitNot(InstructionType) = 0;
  virtual void VisitReshape(InstructionType) = 0;
  virtual void VisitReverse(InstructionType) = 0;
  virtual void VisitRsqrt(InstructionType) = 0;
  virtual void VisitSlice(InstructionType) = 0;
  virtual void VisitSqrt(InstructionType) = 0;
  virtual void VisitTranspose(InstructionType) = 0;

  // Binary
  virtual void VisitAdd(InstructionType) = 0;
  virtual void VisitAnd(InstructionType) = 0;
  virtual void VisitCompare(InstructionType) = 0;
  virtual void VisitDivide(InstructionType) = 0;
  virtual void VisitMaximum(InstructionType) = 0;
  virtual void VisitMinimum(InstructionType) = 0;
  virtual void VisitMultiply(InstructionType) = 0;
  virtual void VisitOr(InstructionType) = 0;
  virtual void VisitSubtract(InstructionType) = 0;
  virtual void VisitXor(InstructionType) = 0;

  // other
  virtual void VisitSelect(InstructionType) = 0;
  virtual void VisitConcatenate(InstructionType) = 0;
  virtual void VisitReduce(InstructionType) = 0;
  virtual void VisitRng(InstructionType) = 0;
  virtual void VisitSort(InstructionType) = 0;
  virtual void VisitTuple(InstructionType) = 0;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
