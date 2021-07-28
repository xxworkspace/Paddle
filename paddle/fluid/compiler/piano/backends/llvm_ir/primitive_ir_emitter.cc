// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "primitive_ir_emitter.h"

namespace piano {

Status PrimitiveIrEmitter::Visit(const NoteInstruction* note) {
#define SWITCH(note_op_code, note)    \
    case NoteOpCode::k##note_op_code: \
        return Handle##note_op_code(note);

    switch(note->OpCode()) {
        SWITCH(Convolution, note);
        //SWITCH(Pooling, note);
        //SWITCH(PoolingGrad, note);
        SWITCH(Dot, note);
        SWITCH(BatchNormalzationInference, note);
        SWITCH(BatchNormalzationTraining, note);
        SWITCH(BatchNormGrad, note);
        SWITCH(Cast, note);
        SWITCH(Exp, note);
        SWITCH(Log, note);
        SWITCH(Negative, note);
        SWITCH(Reverse, note);
        SWITCH(Rsqrt, note);
        SWITCH(Sqrt, note);
        SWITCH(Add, note);
        SWITCH(Subtract, note);
        SWITCH(Multiply, note);
        SWITCH(Divide, note);
        SWITCH(Minimum, note);
        SWITCH(Maximum, note);
        SWITCH(Compare, note);
        SWITCH(And, note);
        SWITCH(Or, note);
        SWITCH(Xor, note);
        SWITCH(Broadcast, note);
        SWITCH(Concatenate, note);
        SWITCH(Copy, note);
        SWITCH(Reduce, note);
        SWITCH(Reshape, note);
        SWITCH(Rng, note);
        SWITCH(Select, note);
        SWITCH(Slice, note);
        SWITCH(Transpose, note);
        SWITCH(Tuple, note);
        default:
            break;
    }
#undef SWITCH 
}

Status PrimitiveIrEmitter::HandleCast(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

Status PrimitiveIrEmitter::HandleExp(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

Status PrimitiveIrEmitter::HandleLog(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

Status PrimitiveIrEmitter::HandleNegative(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

Status PrimitiveIrEmitter::HandleNot(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

Status PrimitiveIrEmitter::HandleReverse(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

Status PrimitiveIrEmitter::HandleRsqrt(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

Status PrimitiveIrEmitter::HandleSqrt(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

Status PrimitiveIrEmitter::HandleAdd(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleSubtract(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleMultiply(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleDivide(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleMaximum(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleMiniMum(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleCompare(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleAnd(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleOr(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

Status PrimitiveIrEmitter::HandleXor(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

}