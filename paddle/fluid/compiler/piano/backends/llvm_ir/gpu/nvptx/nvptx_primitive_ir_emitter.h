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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/gpu/gpu_primitive_ir_emitter.h"

namespace piano {
namespace gpu {

class NvptxPrimitiveIrEmitter : public GpuPrimitiveIrEmitter {
public:
    NvptxPrimitiveIrEmitter(){}
    ~NvptxPrimitiveIrEmitter(){}

    std::function<llvm::value*(llvm::value*, llvm::IRBuilder<>*)> 
        GetUnaryyOp(const NoteInstruction*) override;
    std::function<llvm::value*(llvm::value*, llvm::value*, llvm::IRBuilder<>*)> 
        GetBinaryOp(const NoteInstruction*) override;

    //about the base code block,
    llvm::Value* ThreadIdx(llvm::IRBuilder* ir_builder) override;
    llvm::Value* ThreadIdy(llvm::IRBuilder* ir_builder) override;
    llvm::Value* ThreadIdz(llvm::IRBuilder* ir_builder) override;
    llvm::Value* BlockDimx(llvm::IRBuilder* ir_builder) override;
    llvm::Value* BlockDimy(llvm::IRBuilder* ir_builder) override;
    llvm::Value* BlockDimz(llvm::IRBuilder* ir_builder) override;
    llvm::Value* BlockIdx(llvm::IRBuilder* ir_builder) override;
    llvm::Value* BlockIdy(llvm::IRBuilder* ir_builder) override;
    llvm::Value* BlockIdz(llvm::IRBuilder* ir_builder) override;
    void ThreadSync(llvm::IRBuilder* ir_builder) override;
    llvm::Value* Alloca(llvm::IRBuilder* ir_builder) override;
};

}
}