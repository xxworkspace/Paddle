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

    virtual Status HandleElementwiseUnary(const NoteInstruction* mlo) = 0;
    virtual Status HandleElementwiseBinary(const NoteInstruction* mlo) = 0;

    //AI api
    Status HandleConvolution(const NoteInstruction*) override {};
    Status HandlePooling(const NoteInstruction*) override {};
    Status HandlePoolingGrad(const NoteInstruction*) override {};
    Status HandleDot(const NoteInstruction*) override {};
    Status HandleBatchNormalzationTraining(const NoteInstruction*) override {};
    Status HandleBatchNormGrad(const NoteInstruction*) override {};
    Status HandleBatchNormalzationInference(const NoteInstruction*) override {};

    //Unary
    Status HandleCast(const NoteInstruction*);
    Status HandleCopy(const NoteInstruction*);
    Status HandleExp(const NoteInstruction*);
    Status HandleLog(const NoteInstruction*);
    Status HandleSqrt(const NoteInstruction*);
    Status HandleRsqrt(const NoteInstruction*);
    Status HandleNegative(const NoteInstruction*);

    //Binary
    Status HandleAdd(const NoteInstruction*);
    Status HandleSubtract(const NoteInstruction*);
    Status HandleMultiply(const NoteInstruction*);
    Status HandleDivide(const NoteInstruction*);
    Status HandleMaximum(const NoteInstruction*);
    Status HandleMiniMum(const NoteInstruction*);
    Status HandleCompare(const NoteInstruction*);

    //other
    /*
    virtual Status HandleBroadcast(const NoteInstruction*) = 0;
    virtual Status HandleReduce(const NoteInstruction*) = 0;
    virtual Status HandleReshape(const NoteInstruction*) = 0;
    virtual Status HandleRng(const NoteInstruction*) = 0;
    virtual Status HandleSelect(const NoteInstruction*) = 0;
    virtual Status HandleTranspose(const NoteInstruction*) = 0;

    virtual Status HandleConcat(const NoteInstruction*) = 0;
    virtual Status HandleSlice(const NoteInstruction*) = 0;
    */

    std::vector<PrimitiveBodyGenerator>& GetBodyGenerators();
protected:
    std::vector<PrimitiveBodyGenerator> body_generators_;
};

}