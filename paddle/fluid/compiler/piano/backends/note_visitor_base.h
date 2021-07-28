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

#include "status.h"

namespace piano {

/*
// 根据ResNet50模型训练所使用的算子进行低层IR指令选择enum class MetaOpCode {
     
     Constant,  
     // 复杂算子  Convolution,  Dot,  BatchNormInference,  BatchNormTraining,  BatchNormGrad,  
     // 一元算子  Cast,  Log,  Exp,  Rsqrt,  Negative,  Sqrt,  Reverse,  Not,  
     // 二元算子  Add,  Subtract,  Multiply,  Divide,  Minimum,  Maximum,  Compare,  And,  Or,  Xor,  
     // 其他元算子  Copy, Broadcast,  Reshape,  Rng,  Slice,  Concatenate,  Transpose,  Reduce,  Select,  Sort,  Tuple,};

*/

template<class InstructionPtr>
class NoteVisitorBase {
public:
    NoteVisitorBase(){}
    virtual  ~NoteVisitorBase(){}

    virtual Status Visit(InstructionPtr) = 0;
    //AI api
    virtual Status HandleConvolution(InstructionPtr) = 0;
    //virtual Status HandlePooling(InstructionPtr) = 0;
    //virtual Status HandlePoolingGrad(InstructionPtr*) = 0;
    virtual Status HandleDot(InstructionPtr) = 0;
    virtual Status HandleBatchNormalzationTraining(InstructionPtr) = 0;
    virtual Status HandleBatchNormGrad(InstructionPtr) = 0;
    virtual Status HandleBatchNormalzationInference(InstructionPtr) = 0;

    //Unary
    virtual Status HandleCast(InstructionPtr) = 0;
    virtual Status HandleExp(InstructionPtr) = 0;
    virtual Status HandleLog(InstructionPtr) = 0;
    virtual Status HandleNegative(InstructionPtr) = 0;
    virtual Status HandleNot(InstructionPtr) = 0;
    virtual Status HandleReverse(InstructionPtr) = 0;
    virtual Status HandleRsqrt(InstructionPtr) = 0;
    virtual Status HandleSqrt(InstructionPtr) = 0;

    //Binary
    virtual Status HandleAdd(InstructionPtr) = 0;
    virtual Status HandleSubtract(InstructionPtr) = 0;
    virtual Status HandleMultiply(InstructionPtr) = 0;
    virtual Status HandleDivide(InstructionPtr) = 0;
    virtual Status HandleMaximum(InstructionPtr) = 0;
    virtual Status HandleMiniMum(InstructionPtr) = 0;
    virtual Status HandleCompare(InstructionPtr) = 0;
    virtual Status HandleAnd(InstructionPtr) = 0;
    virtual Status HandleOr(InstructionPtr) = 0;
    virtual Status HandleXor(InstructionPtr) = 0;

    //other
    virtual Status HandleBroadcast(InstructionPtr) = 0;
    virtual Status HandleConcatenate(InstructionPtr) = 0;
    virtual Status HandleCopy(InstructionPtr) = 0;
    virtual Status HandleReduce(InstructionPtr) = 0;
    virtual Status HandleReshape(InstructionPtr) = 0;
    virtual Status HandleRng(InstructionPtr) = 0;
    virtual Status HandleSelect(InstructionPtr) = 0;
    virtual Status HandleSlice(InstructionPtr) = 0;
    virtual Status HandleTranspose(InstructionPtr) = 0;
    virtual Status HandleTuple(InstructionPtr) = 0;
};

}