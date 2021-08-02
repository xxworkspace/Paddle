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

namespace paddle {
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

    virtual void Visit(InstructionPtr) = 0;
    //AI api
    virtual void HandleConvolution(InstructionPtr) = 0;
    //virtual void HandlePooling(InstructionPtr) = 0;
    //virtual void HandlePoolingGrad(InstructionPtr*) = 0;
    virtual void HandleDot(InstructionPtr) = 0;
    virtual void HandleBatchNormalzationTraining(InstructionPtr) = 0;
    virtual void HandleBatchNormGrad(InstructionPtr) = 0;
    virtual void HandleBatchNormalzationInference(InstructionPtr) = 0;

    //Unary
    virtual void HandleCast(InstructionPtr) = 0;
    virtual void HandleExp(InstructionPtr) = 0;
    virtual void HandleLog(InstructionPtr) = 0;
    virtual void HandleNegative(InstructionPtr) = 0;
    virtual void HandleNot(InstructionPtr) = 0;
    virtual void HandleReverse(InstructionPtr) = 0;
    virtual void HandleRsqrt(InstructionPtr) = 0;
    virtual void HandleSqrt(InstructionPtr) = 0;

    //Binary
    virtual void HandleAdd(InstructionPtr) = 0;
    virtual void HandleSubtract(InstructionPtr) = 0;
    virtual void HandleMultiply(InstructionPtr) = 0;
    virtual void HandleDivide(InstructionPtr) = 0;
    virtual void HandleMaximum(InstructionPtr) = 0;
    virtual void HandleMiniMum(InstructionPtr) = 0;
    virtual void HandleCompare(InstructionPtr) = 0;
    virtual void HandleAnd(InstructionPtr) = 0;
    virtual void HandleOr(InstructionPtr) = 0;
    virtual void HandleXor(InstructionPtr) = 0;

    //other
    virtual void HandleBroadcast(InstructionPtr) = 0;
    virtual void HandleConcatenate(InstructionPtr) = 0;
    virtual void HandleCopy(InstructionPtr) = 0;
    virtual void HandleReduce(InstructionPtr) = 0;
    virtual void HandleReshape(InstructionPtr) = 0;
    virtual void HandleRng(InstructionPtr) = 0;
    virtual void HandleSelect(InstructionPtr) = 0;
    virtual void HandleSlice(InstructionPtr) = 0;
    virtual void HandleTranspose(InstructionPtr) = 0;
    virtual void HandleTuple(InstructionPtr) = 0;
};

}
}