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

#include "paddle/fluid/compiler/piano/backends/llvmir/primitive_ir_emitter.h"

namespace paddle {
namespace piano {

void PrimitiveIrEmitter::VisitCast(const note::Instruction* note) {
  VisitElementwiseUnary(note);
}

void PrimitiveIrEmitter::VisitExp(const note::Instruction* note) {
  VisitElementwiseUnary(note);
}

void PrimitiveIrEmitter::VisitLog(const note::Instruction* note) {
  VisitElementwiseUnary(note);
}

void PrimitiveIrEmitter::VisitNegative(const note::Instruction* note) {
  VisitElementwiseUnary(note);
}

void PrimitiveIrEmitter::VisitNot(const note::Instruction* note) {
  VisitElementwiseUnary(note);
}

void PrimitiveIrEmitter::VisitRsqrt(const note::Instruction* note) {
  VisitElementwiseUnary(note);
}

void PrimitiveIrEmitter::VisitSqrt(const note::Instruction* note) {
  VisitElementwiseUnary(note);
}

void PrimitiveIrEmitter::VisitAdd(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitAnd(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitCompare(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitDivide(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitMaximum(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitMinimum(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitMultiply(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitOr(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitSubtract(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

void PrimitiveIrEmitter::VisitXor(const note::Instruction* note) {
  VisitElementwiseBinary(note);
}

std::vector<PrimitiveIrGenerator>
PrimitiveIrEmitter::GetPrimitiveIrGenerators() {
  return primitive_ir_generators_;
}

}  // namespace piano
}  // namespace paddle
