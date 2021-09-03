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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/gpu_ir_emitter.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/llvm_utils.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_primitive_ir_emitter.h"
#include "paddle/fluid/compiler/piano/note/function.h"

namespace paddle {
namespace piano {
namespace backends {

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitElementwiseUnary(
    const note::Instruction& instr) {}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitElementwiseBinary(
    const note::Instruction& instr) {
  // Step 1: Create kernel function, for binary op, it has 4 args,
  // e.g., add(float*, float*, float*, int)
  auto lhs_type = instr.operand(0).shape().element_type();
  auto rhs_type = instr.operand(1).shape().element_type();
  auto out_type = instr.shape().element_type();
  PADDLE_ENFORCE_EQ(
      lhs_type, rhs_type,
      platform::errors::InvalidArgument(
          "The inputs of Binary Op should have the same data type, "
          "but received the types of inputs are %s and %s.",
          lhs_type, rhs_type));

  uint32_t element_count = 1;
  for (auto dim : instr.shape().dimensions()) {
    element_count *= dim;
  }

  auto func =
      CreateLLVMFunction(instr.name(), {lhs_type, rhs_type, out_type},
                         IrEmitter<PrimitiveIrEmitterType>::llvm_module_);

  // Step 2: Create a BasicBlock "entry"
  auto& context = IrEmitter<PrimitiveIrEmitterType>::llvm_module_->getContext();
  auto entry_block = llvm::BasicBlock::Create(context, "entry", func);
  llvm::IRBuilder<> entry_irbuilder(entry_block);

  auto args_it = func->arg_begin();
  llvm::Value* lhs = args_it++;
  llvm::Value* rhs = args_it++;
  llvm::Value* out = args_it++;

  PrimitiveIrEmitterType primitive_ir_emitter(&context, func);

  // Get thread id and this part will be added to BasicBlock "entry"
  auto tidx = primitive_ir_emitter.ThreadIdx(&entry_irbuilder);
  auto bidx = primitive_ir_emitter.BlockIdx(&entry_irbuilder);
  auto block_dim = primitive_ir_emitter.BlockDimx(&entry_irbuilder);
  auto grid_dim = primitive_ir_emitter.GridDimx(&entry_irbuilder);
  auto stride =
      primitive_ir_emitter.Multiply(block_dim, grid_dim, &entry_irbuilder);
  auto index = primitive_ir_emitter.Add(
      primitive_ir_emitter.Multiply(bidx, block_dim, &entry_irbuilder), tidx,
      &entry_irbuilder);

  // Step 3: Get the body_generators for the function. There are 3
  // generators "load", "compute" and "store".
  primitive_ir_emitter.VisitElementwiseBinary(instr);
  auto body_generators = primitive_ir_emitter.GetPrimitiveIrGenerators();
  auto gen_body = [&](llvm::IRBuilder<>* builder) {
    auto vals = body_generators[0].Run(IrArray{lhs, rhs, index}, builder);
    auto res = body_generators[1].Run(vals, builder);
    body_generators[2].Run(IrArray{res[0], out, index}, builder);
  };

  // Step 4: Combine the for loop and the computation to form the function.
  // PrimitiveIrEmitter::For create 3 BasicBlocks, for_begin, for_body and
  // for_end. The gen_body generates ir in for_body. The function returns
  // void in for_end.
  primitive_ir_emitter.For(index, num, stride, gen_body, &entry_irbuilder);
  entry_irbuilder.CreateRetVoid();

  // Step 5: Marking the function as kernel
  llvm::NamedMDNode* nvvm_annotations_node =
      IrEmitter<PrimitiveIrEmitterType>::llvm_module_->getOrInsertNamedMetadata(
          "nvvm.annotations");
  nvvm_annotations_node->addOperand(llvm::MDNode::get(
      context, {llvm::ConstantAsMetadata::get(func),
                llvm::MDString::get(context, "kernel"),
                llvm::ConstantAsMetadata::get(entry_irbuilder.getInt32(1))}));
}

// Scalar op
void GpuIrEmitter::VisitConstant(const note::Instruction& instr) {
  // Constant Instruction Do Nothing.
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitParameter(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Parameter( is unimplemented!"));
}

// Unary
template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitBroadcast(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Broadcast is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitCopy(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Copy is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitReshape(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Reshape is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitReverse(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Reverse is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitSlice(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Slice is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitTranspose(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Transpose is unimplemented!"));
}

// Other
template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitSelect(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Select is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitConcatenate(
    const note::Instruction& instr) {
  PADDLE_THROW(
      platform::errors::Unimplemented("Concatenate is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitReduce(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Reduce is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitRng(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Rng is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitSort(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Sort is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitTuple(
    const note::Instruction& instr) {
  PADDLE_THROW(platform::errors::Unimplemented("Tuple is unimplemented!"));
}

template <typename PrimitiveIrEmitterType>
void GpuIrEmitter<PrimitiveIrEmitterType>::VisitFusion(
    const note::Instruction& instr) {
  // get the function to call
  auto func_ptr = instr.call_functions()[0];
  // build Instruction map for fast retrieval
  std::unordered_map<std::string, note::Instruction*> instrs_map;
  for (note::Instruction& tmp_instr : func_ptr->instructions()) {
    instrs_map[tmp_instr.name()] = &tmp_instr;
  }

  // get the return instruction and output
  const note::Instruction& return_instr = func_ptr->return_instr();
  // build return Instruction map
  std::unordered_map<std::string, const note::Instruction*> return_instrs;
  if (return_instr.opcode() == note::OpCode::kTuple) {
    for (auto op : return_instr.operands()) {
      return_instrs[op->name()] = op;
    }
  } else {
    return_instrs[return_instr.name()] = &return_instr;
  }

  // get function args
  std::vector<note::ElementTypeProto> args_type;
  for (auto op : instr.operands()) {
    args_type.push_back(op->shape().element_type());
  }
  if (return_instr.opcode() == note::OpCode::kTuple) {
    for (auto op : return_instr.operands()) {
      args_type.push_back(op->shape().element_type());
    }
  } else {
    args_type.push_back(return_instr.shape().element_type());
  }

  // create function
  auto func = CreateLLVMFunction(
      instr.name(), args_type, IrEmitter<PrimitiveIrEmitterType>::llvm_module_);
  auto& context = IrEmitter<PrimitiveIrEmitterType>::llvm_module_->getContext();
  auto entry_block = llvm::BasicBlock::Create(context, "entry", func);
  // auto exit_block = llvm::BasicBlock::Create(context, "exit", func);
  llvm::IRBuilder<> entry_irbuilder(entry_block);

  // get input args
  auto args_it = func->arg_begin();
  std::unordered_map<std::string, llvm::Value*> key_values;
  for (auto op : instr.operands()) {
    key_values[op->name()] = args_it++;
  }

  if (return_instr.opcode() == note::OpCode::kTuple) {
    for (auto op : return_instr.operands()) {
      key_values[op->name()] = args_it++;
    }
  } else {
    key_values[return_instr.name()] = args_it++;
  }

  // get body gengerator
  PrimitiveIrEmitterType primitive_ir_emitter(&context, func);
  std::unordered_map<std::string, std::vector<PrimitiveIrGenerator>>
      primitive_ir_generators;
  for (auto element : instrs_map) {
    element.second->Accept(&primitive_ir_emitter);
    primitive_ir_generators[element.first] =
        primitive_ir_emitter.GetPrimitiveIrGenerators();
    primitive_ir_emitter.Clear();
  }

  // TODO(sunli) : sort the instruction by order
  // auto instrs_in_order = Sort(instrs_map);
  llvm::Value* t_index =
      primitive_ir_emitter.GetGridThreadIndex(&entry_irbuilder);

  // TODO(sunli) : check boundry
  // ....

  std::vector<note::Instruction*> instrs_in_order;
  for (auto tmp_instr : instrs_in_order) {
    auto ir_generator = primitive_ir_generators[tmp_instr->name()];
    if (ir_generator.size() == 0) {
      // TODO(sunli) : it's not clear now
      // others instruction need for handle
      continue;
    }

    // get input
    IrArray input_array;
    for (auto op : tmp_instr->operands()) {
      if (key_values.count(op->name() + "_cached") > 0) {
        input_array.push_back(key_values[op->name() + "_cached"]);
      } else {
        // TODO(sunli): load value
        IrArray ir_array = {t_index, key_values[op->name()]};
        input_array.push_back(
            ir_generator[0].Run(ir_array, &entry_irbuilder)[0]);
      }
    }
    // compute
    auto output = ir_generator[1].Run(input_array, &entry_irbuilder);
    // store
    if (return_instrs.count(tmp_instr->name()) > 0) {
      IrArray output_array = {output[0], key_values[tmp_instr->name()],
                              t_index};
      ir_generator[2](output_array, &entry_irbuilder);
    }

    // cache
    key_values[tmp_instr->name() + "_cached"] = output[0];
  }
}

template class GpuIrEmitter<NvptxPrimitiveIrEmitter>;

}  // namespace backends
}  // namespace piano
}  // namespace paddle
