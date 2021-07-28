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

#include "nvptx_primitive_ir_emitter.h"

namespace piano {
namespace gpu {

std::function<llvm::value*(llvm::value*, llvm::IRBuilder<>*)> 
    NvptxPrimitiveIrEmitter::GetUnaryyOp(const NoteInstruction* note) {
    return [](llvm::value* value, llvm::IRBuilder<>* ir_builder) -> llvm::value*{
        return nullptr; 
    };
}

std::function<llvm::value*(llvm::value*, llvm::value*, llvm::IRBuilder<>*)> 
    NvptxPrimitiveIrEmitter::GetBinaryOp(const NoteInstruction* note) {
    return [](llvm::value* first, llvm::value* dst, llvm::IRBuilder<>* ir_builder) -> llvm::value*{
        return nullptr;
    };
}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdx(llvm::IRBuilder* ir_builder) {

}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdy(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdz(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimx(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimy(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimz(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdx(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdy(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdz(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadSync(llvm::IRBuilder* ir_builder) {
    
}

llvm::Value* NvptxPrimitiveIrEmitter::Alloca(llvm::IRBuilder* ir_builder) {
    
}

}
}