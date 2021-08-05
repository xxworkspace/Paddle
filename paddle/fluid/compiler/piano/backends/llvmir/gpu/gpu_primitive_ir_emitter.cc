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

#include "paddle/fluid/compiler/piano/backends/llvmir/gpu/gpu_primitive_ir_emitter.h"

namespace paddle {
namespace piano {
namespace gpu {

void GpuPrimitiveIrEmitter::VisitElementwiseUnary(const NoteInstruction* note) {
}

void GpuPrimitiveIrEmitter::VisitElementwiseBinary(
    const NoteInstruction* note) {}

// Unary
void GpuPrimitiveIrEmitter::VisitBroadcast(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitCopy(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitReshape(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitReverse(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitSlice(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitTranspose(const NoteInstruction* note) {}

// Other
void GpuPrimitiveIrEmitter::VisitSelect(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitConcatenate(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitReduce(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitRng(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitSort(const NoteInstruction* note) {}
void GpuPrimitiveIrEmitter::VisitTuple(const NoteInstruction* note) {}

}  // namespace gpu
}  // namespace piano
}  // namespace paddle
