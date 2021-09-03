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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_primitive_ir_emitter.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_utils.h"

namespace paddle {
namespace piano {
namespace backends {

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdx(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_x;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdy(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_y;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::ThreadIdz(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_tid_z;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimx(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_x;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimy(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_y;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockDimz(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ntid_z;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdx(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_x;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdy(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_y;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::BlockIdz(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_ctaid_z;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::GridDimx(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_x;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::GridDimy(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_y;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

llvm::Value* NvptxPrimitiveIrEmitter::GridDimz(llvm::IRBuilder<>* ir_builder) {
  llvm::Intrinsic::ID llvm_Intrinsic =
      llvm::Intrinsic::nvvm_read_ptx_sreg_nctaid_z;
  return CallToLLVMIntrinsic(ir_builder, llvm_Intrinsic);
}

void NvptxPrimitiveIrEmitter::ThreadSync(llvm::IRBuilder<>* ir_builder) {}

llvm::Value* NvptxPrimitiveIrEmitter::Alloca(llvm::IRBuilder<>* ir_builder,
                                             unsigned) {
  return nullptr;
}

llvm::Value* NvptxPrimitiveIrEmitter::GetGridThreadIndex(
    llvm::IRBuilder<>* ir_builder) {
  // (blockIdx.y*blockDim.y + threadIdx.y) * (blockDim.x * gridDim.x) +
  // blockDim.x * blockIdx.x + threadIdx.x;
  auto row_size =
      ir_builder->CreateMul(BlockDimx(ir_builder), GridDimx(ir_builder));
  auto col = ir_builder->CreateAdd(
      ir_builder->CreateMul(BlockIdy(ir_builder), BlockDimy(ir_builder)),
      ThreadIdy(ir_builder));
  return ir_builder->CreateAdd(
      ir_builder->CreateAdd(
          ir_builder->CreateMul(row_size, col),
          ir_builder->CreateMul(BlockDimx(ir_builder), BlockIdx(ir_builder))),
      ThreadIdx(ir_builder));
}

llvm::Value* NvptxPrimitiveIrEmitter::GetBlockThreadIndex(
    llvm::IRBuilder<>* ir_builder) {
  // blockDim.x * blockIdx.y + threadIdx.x
  return ir_builder->CreateAdd(
      ir_builder->CreateMul(BlockDimx(ir_builder), BlockIdy(ir_builder)),
      ThreadIdx(ir_builder));
}

llvm::Value* NvptxPrimitiveIrEmitter::GetWarpThreadIndex(llvm::IRBuilder<>*) {
  return nullptr;
}

llvm::Value* NvptxPrimitiveIrEmitter::GetGridBlockIndex(llvm::IRBuilder<>*) {
  return nullptr;
}

llvm::Value* NvptxPrimitiveIrEmitter::GetBlockWarpIndex(llvm::IRBuilder<>*) {
  return nullptr;
}

llvm::Value* NvptxPrimitiveIrEmitter::GetBlockSize(
    llvm::IRBuilder<>* ir_builder) {
  return ir_builder->CreateMul(BlockDimx(ir_builder), BlockDimy(ir_builder));
}

llvm::Value* NvptxPrimitiveIrEmitter::GetThreadCount(
    llvm::IRBuilder<>* ir_builder) {
  // GridDimx * blockDimx * GridDimy * BlockDimy
  return ir_builder->CreateMul(
      ir_builder->CreateMul(BlockDimy(ir_builder), GridDimy(ir_builder)),
      ir_builder->CreateMul(BlockDimx(ir_builder), GridDimx(ir_builder)));
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
