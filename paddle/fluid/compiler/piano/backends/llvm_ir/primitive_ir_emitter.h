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

#include <vector>
#include "llvm/IR/Module.h"
#include "paddle/fluid/compiler/piano/backends/note_visitor_base.h"
#include "primitive_body_generator.h"

namespace piano {

class PrimitiveIrEmitter : public MloVisitorBase<const NoteInstruction*> {
public:
    PrimitiveIrEmitter(){}
    virtual ~PrimitiveIrEmitter(){}

    Status Visit(const NoteInstruction*) override;

    virtual Status HandleElementwiseUnary(const NoteInstruction*) = 0;
    virtual Status HandleElementwiseBinary(const NoteInstruction*) = 0;

    //AI api
    Status HandleConvolution(const NoteInstruction*) override {};
    //virtual Status HandlePooling(InstructionPtr*) = 0;
    //virtual Status HandlePoolingGrad(InstructionPtr*) = 0;
    Status HandleDot(const NoteInstruction*) override {};
    Status HandleBatchNormalzationTraining(const NoteInstruction*) override {};
    Status HandleBatchNormGrad(const NoteInstruction*) override {};
    Status HandleBatchNormalzationInference(const NoteInstruction*) override {};

    //Unary
    virtual Status HandleCast(const NoteInstruction*);
    virtual Status HandleExp(const NoteInstruction*);
    virtual Status HandleLog(const NoteInstruction*);
    virtual Status HandleNegative(const NoteInstruction*);
    virtual Status HandleNot(const NoteInstruction*);
    virtual Status HandleReverse(const NoteInstruction*);
    virtual Status HandleRsqrt(const NoteInstruction*);
    virtual Status HandleSqrt(const NoteInstruction*);
 
    //Binary
    virtual Status HandleAdd(const NoteInstruction*);
    virtual Status HandleSubtract(const NoteInstruction*);
    virtual Status HandleMultiply(const NoteInstruction*);
    virtual Status HandleDivide(const NoteInstruction*);
    virtual Status HandleMaximum(const NoteInstruction*);
    virtual Status HandleMiniMum(const NoteInstruction*);
    virtual Status HandleCompare(const NoteInstruction*);
    virtual Status HandleAnd(const NoteInstruction*);
    virtual Status HandleOr(const NoteInstruction*);
    virtual Status HandleXor(const NoteInstruction*);

    //other
    virtual Status HandleBroadcast(const NoteInstruction*) {};
    virtual Status HandleConcatenate(const NoteInstruction*) {};
    virtual Status HandleCopy(const NoteInstruction*) {};
    virtual Status HandleReduce(const NoteInstruction*) {};
    virtual Status HandleReshape(const NoteInstruction*) {};
    virtual Status HandleRng(const NoteInstruction*) {};
    virtual Status HandleSelect(const NoteInstruction*) {};
    virtual Status HandleSlice(const NoteInstruction*) {};
    virtual Status HandleTranspose(const NoteInstruction*) {};
    virtual Status HandleTuple(const NoteInstruction*) {};

    std::vector<PrimitiveBodyGenerator>& GetBodyGenerators();
protected:
    std::vector<PrimitiveBodyGenerator> body_generators_;
};

}