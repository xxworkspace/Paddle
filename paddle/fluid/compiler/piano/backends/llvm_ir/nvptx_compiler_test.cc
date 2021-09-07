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

#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_compiler.h"
#include "glog/logging.h"
#include "gtest/gtest.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/executable_pool.h"
#include "paddle/fluid/compiler/piano/backends/llvm_ir/nvptx_executable.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/symbolization/note_builder.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace piano {
namespace backends {

class TestCompiler : public NvptxCompiler {
 public:
  std::string CallCompileToPtx(llvm::Module* llvm_module) {
    return CompileToPtx(llvm_module);
  }

  std::string CallGetLlvmTriple() { return GetLlvmTriple(); }

  std::string CallGetLlvmDataLayout() { return GetLlvmDataLayout(); }
};

TEST(NvptxCompiler, CompileToPtx) {
  TestCompiler test_compiler;
  // llvm::IR
  llvm::LLVMContext context;
  llvm::Module llvm_module("CuAdd_Test", context);

  llvm_module.setTargetTriple(
      llvm::StringRef(test_compiler.CallGetLlvmTriple()));
  llvm_module.setDataLayout(
      llvm::StringRef(test_compiler.CallGetLlvmDataLayout()));

  {
    llvm::SmallVector<llvm::Type*, 4> args;
    args.push_back(llvm::Type::getFloatPtrTy(context, 1));
    args.push_back(llvm::Type::getFloatPtrTy(context, 1));
    args.push_back(llvm::Type::getFloatPtrTy(context, 1));
    // args.push_back(llvm::Type::getInt32Ty(context));
    llvm::Type* retType = llvm::Type::getVoidTy(context);

    llvm::FunctionType* addType = llvm::FunctionType::get(retType, args, false);
    llvm_module.getOrInsertFunction("CuAdd", addType);
    auto cu_add = llvm_module.getFunction("CuAdd");

    auto args_it = cu_add->arg_begin();
    llvm::Value* arg_a = args_it++;
    llvm::Value* arg_b = args_it++;
    llvm::Value* arg_c = args_it++;
    // llvm::Value* arg_n = args_it ++;

    auto entry = llvm::BasicBlock::Create(context, "entry", cu_add);
    llvm::IRBuilder<> entry_builder(entry);
    llvm::IRBuilder<> add_builder(entry);
    auto tidx = llvm::Intrinsic::getDeclaration(
        &llvm_module,
        llvm::Intrinsic::NVVMIntrinsics::nvvm_read_ptx_sreg_tid_x);
    auto idx = entry_builder.CreateCall(tidx);
    auto _1 = entry_builder.CreateGEP(arg_a, idx);
    auto _2 = entry_builder.CreateGEP(arg_b, idx);
    auto _3 = entry_builder.CreateGEP(arg_c, idx);

    auto _4 = entry_builder.CreateLoad(_1);
    auto _5 = entry_builder.CreateLoad(_2);
    auto _6 = entry_builder.CreateFAdd(_4, _5);
    entry_builder.CreateStore(_6, _3);

    entry_builder.CreateRetVoid();

    llvm::NamedMDNode* nvvm_annotations_node =
        llvm_module.getOrInsertNamedMetadata("nvvm.annotations");
    nvvm_annotations_node->addOperand(llvm::MDNode::get(
        context, {llvm::ConstantAsMetadata::get(cu_add),
                  llvm::MDString::get(context, "kernel"),
                  llvm::ConstantAsMetadata::get(entry_builder.getInt32(1))}));
  }
  llvm_module.print(llvm::errs(), nullptr);
  auto ptx = test_compiler.CallCompileToPtx(&llvm_module);
  LOG(INFO) << ptx;

  // set device
  platform::SetDeviceId(0);
  CumodulePool::Instance().Insert("CompileToPtx", ptx);
  auto func = CumodulePool::Instance().GetCuFunction("CompileToPtx", "CuAdd");

  float host_a[32], host_b[32], host_c[32];
  for (int idx = 0; idx < 32; ++idx) {
    host_a[idx] = static_cast<float>(idx);
    host_b[idx] = static_cast<float>(idx);
  }

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_a, 32 * sizeof(float)));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_b, 32 * sizeof(float)));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_c, 32 * sizeof(float)));

  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(dev_a, host_a, sizeof(float) * 32, cudaMemcpyHostToDevice));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(dev_b, host_b, sizeof(float) * 32, cudaMemcpyHostToDevice));

  std::vector<void*> args = {&dev_a, &dev_b, &dev_c};

  CHECK_CUDA_DRIVER_SUCCESS(platform::dynload::cuLaunchKernel(
      func, 1, 1, 1, 32, 1, 1, 0, nullptr, args.data(), nullptr));

  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(host_c, dev_c, sizeof(float) * 32, cudaMemcpyDeviceToHost));

  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_a));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_b));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_c));

  for (int idx = 0; idx < 32; ++idx) {
    ASSERT_EQ(host_c[idx], 2.0 * idx);
  }
}

TEST(NvptxCompiler, Apply) {
  // set device
  platform::SetDeviceId(0);

  // note builder
  symbolization::NoteBuilder note_builder("test_note_builder");
  {
    note::InstructionProto a_proto, b_proto, c_proto;
    a_proto.set_name("A");
    a_proto.set_opcode("constant");
    auto a_shape = a_proto.mutable_shape();
    a_shape->set_element_type(note::ElementTypeProto::F32);
    a_shape->add_dimensions(1);
    a_shape->add_dimensions(32);

    b_proto.set_name("B");
    b_proto.set_opcode("constant");
    auto b_shape = b_proto.mutable_shape();
    b_shape->set_element_type(note::ElementTypeProto::F32);
    b_shape->add_dimensions(1);
    b_shape->add_dimensions(32);

    c_proto.set_name("C");
    c_proto.set_opcode("add");
    auto c_shape = c_proto.mutable_shape();
    c_shape->set_element_type(note::ElementTypeProto::F32);
    c_shape->add_dimensions(1);
    c_shape->add_dimensions(32);

    // build note module
    std::vector<symbolization::Operand> ops;
    ops.push_back(note_builder.AppendInstruction(std::move(a_proto),
                                                 note::OpCode::kConstant, {}));
    ops.push_back(note_builder.AppendInstruction(std::move(b_proto),
                                                 note::OpCode::kConstant, {}));
    note_builder.AppendInstruction(std::move(c_proto), note::OpCode::kAdd, ops);
  }
  auto note_proto = note_builder.Build();
  note::Module note_module(note_proto);

  // compile
  NvptxCompiler nvptx_compiler;
  // To removet 'EXPECT_THROW' in future commit
  // EXPECT_THROW(nvptx_compiler.Apply(&note_module), platform::EnforceNotMet);
  nvptx_compiler.Apply(&note_module);

  // TestExecutable
  auto* instr = note_module.entry_function().instruction(std::int64_t(2));
  NvptxExecutable executable(note_module.name(), dim3(1), dim3(32), 0, *instr);

  float host_a[32], host_b[32], host_c[32];
  for (int idx = 0; idx < 32; ++idx) {
    host_a[idx] = static_cast<float>(idx);
    host_b[idx] = static_cast<float>(idx);
  }

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr;
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_a, 32 * sizeof(float)));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_b, 32 * sizeof(float)));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_c, 32 * sizeof(float)));

  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(dev_a, host_a, sizeof(float) * 32, cudaMemcpyHostToDevice));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(dev_b, host_b, sizeof(float) * 32, cudaMemcpyHostToDevice));

  std::vector<void*> args = {&dev_a, &dev_b, &dev_c};
  executable.Launch(args, nullptr);

  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(host_c, dev_c, sizeof(float) * 32, cudaMemcpyDeviceToHost));

  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_a));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_b));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_c));

  for (int idx = 0; idx < 32; ++idx) {
    ASSERT_EQ(host_c[idx], 2.0 * idx);
  }
}

TEST(NvptxCompiler, Fusion) {
  note::ModuleProto module_proto;
  {
    // module
    module_proto.set_name("test_builder");
    module_proto.set_entry_function_name("entry_func");
    module_proto.set_id(0);
    module_proto.set_entry_function_id(10);
    // fusion function
    {
      auto func = module_proto.add_functions();
      func->set_name("fusion_func");
      func->set_id(11);
      func->set_return_id(111);

      {
        auto param = func->add_instructions();
        param->set_name("parameter0");
        param->set_opcode("parameter");
        auto shape = param->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        param->set_id(105);
        param->set_parameter_index(0);
      }

      {
        auto param = func->add_instructions();
        param->set_name("parameter1");
        param->set_opcode("parameter");
        auto shape = param->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        param->set_id(106);
        param->set_parameter_index(1);
      }

      {
        auto param = func->add_instructions();
        param->set_name("parameter2");
        param->set_opcode("parameter");
        auto shape = param->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        param->set_id(107);
        param->set_parameter_index(2);
      }

      {
        auto param = func->add_instructions();
        param->set_name("parameter3");
        param->set_opcode("parameter");
        auto shape = param->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        param->set_id(108);
        param->set_parameter_index(3);
      }

      {
        auto add = func->add_instructions();
        add->set_name("add0");
        add->set_opcode("add");
        auto shape = add->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        add->set_id(109);
        add->add_operand_ids(105);
        add->add_operand_ids(106);
      }

      {
        auto add = func->add_instructions();
        add->set_name("add1");
        add->set_opcode("add");
        auto shape = add->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        add->set_id(110);
        add->add_operand_ids(107);
        add->add_operand_ids(109);
      }

      {
        auto add = func->add_instructions();
        add->set_name("add2");
        add->set_opcode("add");
        auto shape = add->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        add->set_id(111);
        add->add_operand_ids(108);
        add->add_operand_ids(110);
      }
    }

    {
      // entry function
      auto entry_func = module_proto.add_functions();
      entry_func->set_name("entry_func");
      entry_func->set_id(10);
      entry_func->set_return_id(100);

      // constant 0
      {
        auto constant = entry_func->add_instructions();
        constant->set_name("input0");
        constant->set_opcode("constant");
        auto shape = constant->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        constant->set_id(101);
      }

      // constant 1
      {
        auto constant = entry_func->add_instructions();
        constant->set_name("input1");
        constant->set_opcode("constant");
        auto shape = constant->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        constant->set_id(102);
      }

      // constant 2
      {
        auto constant = entry_func->add_instructions();
        constant->set_name("input2");
        constant->set_opcode("constant");
        auto shape = constant->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        constant->set_id(103);
      }

      // constant 3
      {
        auto constant = entry_func->add_instructions();
        constant->set_name("input3");
        constant->set_opcode("constant");
        auto shape = constant->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        constant->set_id(104);
      }

      // fusion instruction
      {
        auto element_wise = entry_func->add_instructions();
        element_wise->set_name("element_wise");
        element_wise->set_opcode("fusion");
        auto shape = element_wise->mutable_shape();
        shape->set_element_type(note::ElementTypeProto::F32);
        shape->add_dimensions(1);
        shape->add_dimensions(32);
        element_wise->set_id(100);
        element_wise->add_call_function_ids(11);
        element_wise->add_operand_ids(101);
        element_wise->add_operand_ids(102);
        element_wise->add_operand_ids(103);
        element_wise->add_operand_ids(104);
      }
    }
  }
  LOG(INFO) << module_proto.DebugString();

  note::Module note_module(module_proto);
  std::cout << note_module.ToString() << std::endl;
  NvptxCompiler nvptx_compiler;
  nvptx_compiler.Apply(&note_module);

  // set device
  platform::SetDeviceId(0);
  // TestExecutable
  auto&& instrs = note_module.entry_function().instructions();
  note::Instruction* fusion_instr;
  for (auto& _instr : instrs) {
    if (_instr.global_id() == 100) {
      fusion_instr = &_instr;
      break;
    }
  }
  NvptxExecutable executable(note_module.name(), dim3(1), dim3(32), 0,
                             *fusion_instr);

  float host_a[32], host_b[32], host_c[32], host_d[32], host_e[32];
  for (int idx = 0; idx < 32; ++idx) {
    host_a[idx] = static_cast<float>(3.14f);
    host_b[idx] = static_cast<float>(3.14f);
    host_c[idx] = static_cast<float>(3.14f);
    host_d[idx] = static_cast<float>(3.14f);
  }

  float *dev_a = nullptr, *dev_b = nullptr, *dev_c = nullptr, *dev_d = nullptr,
        *dev_e = nullptr;
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_a, 32 * sizeof(float)));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_b, 32 * sizeof(float)));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_c, 32 * sizeof(float)));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_d, 32 * sizeof(float)));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMalloc(&dev_e, 32 * sizeof(float)));

  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(dev_a, host_a, sizeof(float) * 32, cudaMemcpyHostToDevice));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(dev_b, host_b, sizeof(float) * 32, cudaMemcpyHostToDevice));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(dev_c, host_c, sizeof(float) * 32, cudaMemcpyHostToDevice));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(dev_d, host_d, sizeof(float) * 32, cudaMemcpyHostToDevice));

  std::vector<void*> args = {&dev_a, &dev_b, &dev_c, &dev_d, &dev_e};
  executable.Launch(args, nullptr);

  PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(nullptr));
  PADDLE_ENFORCE_CUDA_SUCCESS(
      cudaMemcpy(host_e, dev_e, sizeof(float) * 32, cudaMemcpyDeviceToHost));

  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_a));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_b));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_c));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_d));
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFree(dev_e));

  for (int idx = 0; idx < 32; ++idx) {
    ASSERT_EQ(host_e[idx], 4.0 * 3.14f);
  }
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
