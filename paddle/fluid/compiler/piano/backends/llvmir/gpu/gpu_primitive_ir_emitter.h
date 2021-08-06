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
#pragma once

#include <llvm/IR/IRBuilder.h>
#include "paddle/fluid/compiler/piano/backends/llvmir/primitive_ir_emitter.h"

namespace paddle {
namespace piano {
namespace gpu {

class GpuPrimitiveIrEmitter : public PrimitiveIrEmitter {
 public:
  virtual std::function<llvm::Value*(llvm::Value*, llvm::Value*,
                                     llvm::IRBuilder<>*)>
  GetBinaryOp(const Instruction*) = 0;
  virtual std::function<llvm::Value*(llvm::Value*, llvm::IRBuilder<>*)>
  GetUnaryOp(const Instruction*) = 0;

  void VisitElementwiseUnary(const Instruction*) override;
  void VisitElementwiseBinary(const Instruction*) override;

  // Unary
  void VisitBroadcast(const Instruction*) override;
  void VisitCopy(const Instruction*) override;
  void VisitReshape(const Instruction*) override;
  void VisitReverse(const Instruction*) override;
  void VisitSlice(const Instruction*) override;
  void VisitTranspose(const Instruction*) override;

  // other
  void VisitSelect(const Instruction*) override;
  void VisitConcatenate(const Instruction*) override;
  void VisitReduce(const Instruction*) override;
  void VisitRng(const Instruction*) override;
  void VisitSort(const Instruction*) override;
  void VisitTuple(const Instruction*) override;

  // about the base code block,
  virtual llvm::Value* ThreadIdx(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* ThreadIdy(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* ThreadIdz(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockDimx(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockDimy(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockDimz(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockIdx(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockIdy(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* BlockIdz(llvm::IRBuilder<>*) = 0;
  virtual void ThreadSync(llvm::IRBuilder<>*) = 0;
  virtual llvm::Value* Alloca(llvm::IRBuilder<>*, unsigned) = 0;
};

}  // namespace gpu
}  // namespace piano
}  // namespace paddle
