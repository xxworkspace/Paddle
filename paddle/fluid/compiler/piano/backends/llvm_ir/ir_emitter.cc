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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/ir_emitter.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_primitive_ir_emitter.h"

namespace paddle {
namespace piano {
namespace backends {

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitCast(
    const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitExp(
    const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitLog(
    const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitNegative(
    const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitNot(
    const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitRsqrt(
    const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitSqrt(
    const note::Instruction& instr) {
  VisitElementwiseUnary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitAdd(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitAnd(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitCompare(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitDivide(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitMaximum(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitMinimum(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitMultiply(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitOr(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitSubtract(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template <typename PrimitiveIrEmitterType>
void IrEmitter<PrimitiveIrEmitterType>::VisitXor(
    const note::Instruction& instr) {
  VisitElementwiseBinary(instr);
}

template class IrEmitter<NvptxPrimitiveIrEmitter>;

}  // namespace backends
}  // namespace piano
}  // namespace paddle
