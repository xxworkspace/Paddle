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

#include "gpu_primitive_ir_emitter.h"

namespace piano {
namespace gpu {

Status GpuPrimitiveIrEmitter::HandleElementwiseUnary(const NoteInstruction* note) {
     //read
    body_generators_.empalce_back("Read_", "READ", [this](IrArray& llvm_values, llvm::IRBuilder<>* llvm_builder){
        return IrArray{llvm_builder->CreateLoad(llvm_values[0]), llvm_builder->CreateLoad(llvm_values[1])};
    });

    //compute
    body_generators_.empalce_back("Read_", "COMPUTE", [this, note](IrArray& llvm_values, llvm::IRBuilder<>* llvm_builder){
        return IrArray{GetBinaryOp(note)(llvm_values[0], llvm_values[1], llvm_builder)};
    });

    //store
    body_generators_.empalce_back("Store_", "STORE", [this](IrArray& llvm_values, llvm::IRBuilder<>* llvm_builder){
        return IrArray{llvm_builder->CreateStore(llvm_values[0], llvm_values[1])};
    });

    return Status();
}

Status GpuPrimitiveIrEmitter::HandleElementwiseBinary(const NoteInstruction* note) {
     //read
    body_generators_.empalce_back("Read_", "READ", [this](IrArray& llvm_values, llvm::IRBuilder<>* llvm_builder){
        return IrArray{llvm_builder->CreateLoad(llvm_values[0])};
    });

    //compute
    body_generators_.empalce_back("Read_", "COMPUTE", [this, note](IrArray& llvm_values, llvm::IRBuilder<>* llvm_builder){
        return IrArray{GetUnaryOp(note)(llvm_values[0], llvm_builder)};
    });

    //store
    body_generators_.empalce_back("Store_", "STORE", [this](IrArray& llvm_values, llvm::IRBuilder<>* llvm_builder){
        return IrArray{llvm_builder->CreateStore(llvm_values[0], llvm_values[1])};
    });

    return Status();
}

Status puPrimitiveIrEmitter::HandleBroadcast(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleConcatenate(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleCopy(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleReduce(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleReshape(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleRng(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleSelect(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleSlice(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleTranspose(const NoteInstruction* note) {
    return Status();
}

Status puPrimitiveIrEmitter::HandleTuple(const NoteInstruction* note) {
    return Status();
}

}
}