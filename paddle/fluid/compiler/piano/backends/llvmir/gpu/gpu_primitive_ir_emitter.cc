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

void GpuPrimitiveIrEmitter::VisitElementwiseUnary(const Instruction* note) {}

void GpuPrimitiveIrEmitter::VisitElementwiseBinary(const Instruction* note) {}

// Unary
void GpuPrimitiveIrEmitter::VisitBroadcast(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitCopy(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitReshape(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitReverse(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitSlice(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitTranspose(const Instruction* note) {}

// Other
void GpuPrimitiveIrEmitter::VisitSelect(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitConcatenate(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitReduce(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitRng(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitSort(const Instruction* note) {}
void GpuPrimitiveIrEmitter::VisitTuple(const Instruction* note) {}

}  // namespace gpu
}  // namespace piano
}  // namespace paddle
