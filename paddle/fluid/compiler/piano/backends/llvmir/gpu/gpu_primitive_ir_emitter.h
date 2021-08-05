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
  GetBinaryOp(const NoteInstruction*) = 0;
  virtual std::function<llvm::Value*(llvm::Value*, llvm::IRBuilder<>*)>
  GetUnaryOp(const NoteInstruction*) = 0;

  void VisitElementwiseUnary(const NoteInstruction*) override;
  void VisitElementwiseBinary(const NoteInstruction*) override;

  // Unary
  void VisitBroadcast(const NoteInstruction*) override;
  void VisitCopy(const NoteInstruction*) override;
  void VisitReshape(const NoteInstruction*) override;
  void VisitReverse(const NoteInstruction*) override;
  void VisitSlice(const NoteInstruction*) override;
  void VisitTranspose(const NoteInstruction*) override;

  // other
  void VisitSelect(const NoteInstruction*) override;
  void VisitConcatenate(const NoteInstruction*) override;
  void VisitReduce(const NoteInstruction*) override;
  void VisitRng(const NoteInstruction*) override;
  void VisitSort(const NoteInstruction*) override;
  void VisitTuple(const NoteInstruction*) override;

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
