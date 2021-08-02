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

#include "ir_emitter.h"

namespace paddle {
namespace piano {

void IrEmitter::Visit(const NoteInstruction* note) {
#define SWITCH(note_op_code, note)    \
    case NoteOpCode::k##note_op_code: \
        Handle##note_op_code(note);   \
        break;

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

void IrEmitter::HandleCast(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

void IrEmitter::HandleExp(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

void IrEmitter::HandleLog(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

void IrEmitter::HandleNegative(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

void IrEmitter::HandleNot(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

void IrEmitter::HandleReverse(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

void IrEmitter::HandleRsqrt(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

void IrEmitter::HandleSqrt(const NoteInstruction* note) {
    return HandleElementwiseUnary(note);
}

void IrEmitter::HandleAdd(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleSubtract(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleMultiply(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleDivide(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleMaximum(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleMiniMum(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleCompare(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleAnd(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleOr(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

void IrEmitter::HandleXor(const NoteInstruction* note) {
    return HandleElementwiseBinary(note);
}

}
}