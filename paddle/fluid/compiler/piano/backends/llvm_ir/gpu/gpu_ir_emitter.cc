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
#include "gpu_ir_emitter.h"

namespace piano {
namespace gpu {

Status GpuIrEmitter::HandleElementwiseUnary(const NoteInstruction* note) {
    return Status();
}

Status GpuIrEmitter::HandleElementwiseBinary(const NoteInstruction* note) {
    return Status();
}

Status GpuIrEmitter::HandleBroadcast(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleConcatenate(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleCopy(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleReduce(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleReshape(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleRng(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleSelect(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleSlice(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleTranspose(const NoteInstruction*) {
    return Status();
}

Status GpuIrEmitter::HandleTuple(const NoteInstruction*) {
    return Status();
}

}
}