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

template <typename Type>
class NoteVisitorBase {
 public:
  virtual ~NoteVisitorBase() {}

  // Scalar op
  virtual void VisitConstant(Type) = 0;
  virtual void VisitParameter(Type) = 0;

  // ops can be replaced by library
  virtual void VisitBatchNormGrad(Type) = 0;
  virtual void VisitBatchNormInference(Type) = 0;
  virtual void VisitBatchNormTraining(Type) = 0;
  virtual void VisitConvolution(Type) = 0;
  virtual void VisitDot(Type) = 0;

  // Unary
  virtual void VisitBroadcast(Type) = 0;
  virtual void VisitCast(Type) = 0;
  virtual void VisitCopy(Type) = 0;
  virtual void VisitExp(Type) = 0;
  virtual void VisitLog(Type) = 0;
  virtual void VisitNegative(Type) = 0;
  virtual void VisitNot(Type) = 0;
  virtual void VisitReshape(Type) = 0;
  virtual void VisitReverse(Type) = 0;
  virtual void VisitRsqrt(Type) = 0;
  virtual void VisitSlice(Type) = 0;
  virtual void VisitSqrt(Type) = 0;
  virtual void VisitTranspose(Type) = 0;

  // Binary
  virtual void VisitAdd(Type) = 0;
  virtual void VisitAnd(Type) = 0;
  virtual void VisitCompare(Type) = 0;
  virtual void VisitDivide(Type) = 0;
  virtual void VisitMaximum(Type) = 0;
  virtual void VisitMinimum(Type) = 0;
  virtual void VisitMultiply(Type) = 0;
  virtual void VisitOr(Type) = 0;
  virtual void VisitSubtract(Type) = 0;
  virtual void VisitXor(Type) = 0;

  // other
  virtual void VisitSelect(Type) = 0;
  virtual void VisitConcatenate(Type) = 0;
  virtual void VisitReduce(Type) = 0;
  virtual void VisitRng(Type) = 0;
  virtual void VisitSort(Type) = 0;
  virtual void VisitTuple(Type) = 0;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
