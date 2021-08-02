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

#include "llvm/IR/Module.h"
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"

namespace paddle {
namespace piano {

class IrEmitter : public NoteVisitorBase<const NoteInstruction*> {
public:
    IrEmitter(llvm::Module* llvm_module, ScheduleMap* schedule)
      :llvm_module_(llvm_module), schedules_(schedule){}
    ~IrEmitter(){}

    void Visit(const NoteInstruction*) override;

    virtual void HandleElementwiseUnary(const NoteInstruction*)  = 0;
    virtual void HandleElementwiseBinary(const NoteInstruction*) = 0;

    //AI api
    virtual void HandleConvolution(const NoteInstruction*) {};
    //virtual void HandlePooling(InstructionPtr*) {};
    //virtual void HandlePoolingGrad(InstructionPtr*) {};
    virtual void HandleDot(const NoteInstruction*) {};
    virtual void HandleBatchNormalzationTraining(const NoteInstruction*) {};
    virtual void HandleBatchNormGrad(const NoteInstruction*) {};
    virtual void HandleBatchNormalzationInference(const NoteInstruction*) {};

    //Unary
    virtual void HandleCast(const NoteInstruction*);
    virtual void HandleExp(const NoteInstruction*);
    virtual void HandleLog(const NoteInstruction*);
    virtual void HandleNegative(const NoteInstruction*);
    virtual void HandleNot(const NoteInstruction*);
    virtual void HandleReverse(const NoteInstruction*);
    virtual void HandleRsqrt(const NoteInstruction*);
    virtual void HandleSqrt(const NoteInstruction*);

    //Binary
    virtual void HandleAdd(const NoteInstruction*);
    virtual void HandleSubtract(const NoteInstruction*);
    virtual void HandleMultiply(const NoteInstruction*);
    virtual void HandleDivide(const NoteInstruction*);
    virtual void HandleMaximum(const NoteInstruction*);
    virtual void HandleMiniMum(const NoteInstruction*);
    virtual void HandleCompare(const NoteInstruction*);
    virtual void HandleAnd(const NoteInstruction*);
    virtual void HandleOr(const NoteInstruction*);
    virtual void HandleXor(const NoteInstruction*);

    //other
    virtual void HandleBroadcast(const NoteInstruction*) {};
    virtual void HandleConcatenate(const NoteInstruction*) {};
    virtual void HandleCopy(const NoteInstruction*) {};
    virtual void HandleReduce(const NoteInstruction*) {};
    virtual void HandleReshape(const NoteInstruction*) {};
    virtual void HandleRng(const NoteInstruction*) {};
    virtual void HandleSelect(const NoteInstruction*) {};
    virtual void HandleSlice(const NoteInstruction*) {};
    virtual void HandleTranspose(const NoteInstruction*) {};
    virtual void HandleTuple(const NoteInstruction*) {};

protected:
    llvm::Module* llvm_module_;
    ScheduleMap* schedules_;
};

}
}