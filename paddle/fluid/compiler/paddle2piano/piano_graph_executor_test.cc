/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/compiler/paddle2piano/piano_graph_executor.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <array>

#include "paddle/fluid/compiler/paddle2piano/piano_op_kernel.h"
#include "paddle/fluid/compiler/paddle2piano/piano_op_kernel_context.h"
#include "paddle/fluid/compiler/paddle2piano/piano_op_registry.h"
#include "paddle/fluid/compiler/paddle2piano/vartype_utils.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace piano {

using framework::ir::Node;
using framework::VariableNameMap;
using framework::AttributeMap;
using GraphNodeVec = PianoGraphExecutor::GraphNodeVec;

using paddle::framework::InferShapeContext;
using paddle::framework::OpProtoAndCheckerMaker;
using paddle::framework::OperatorWithKernel;

std::unordered_set<note::ElementTypeProto> TestDatatypes() {
  static std::unordered_set<note::ElementTypeProto> supported_types = {
      note::F16, note::F32, note::F64};
  return supported_types;
}

std::vector<std::string>& GetCompileOrder() {
  static std::vector<std::string> order;
  return order;
}

class TestOp : public OperatorWithKernel {
 public:
  using OperatorWithKernel::OperatorWithKernel;

  void InferShape(InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("X"), "Input", "X", "test");
    OP_INOUT_CHECK(ctx->HasOutput("Out"), "Output", "Out", "test");

    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim("Out", in_dims);
    ctx->ShareLoD("X", /*->*/ "Out");
  }
};

class TestOpMaker : public OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor), The input tensor of test op.");
    AddOutput("Out", "(Tensor), The output tensor of test op.");
    AddComment(R"DOC(Test Operator.)DOC");
  }
};

// register piano op kernel with limit allow backend list
class TestPianoOpMaker : public PianoOpMaker {
 public:
  void Make() override {
    // do nothing, pass
  }
};

class TestPianoOpKernel : public PianoOpKernel {
 public:
  void Compile(const PianoOpKernelContext& context) const override {
    // do nothing, pass
    GetCompileOrder().emplace_back(context.Type());
  }
};

std::array<std::vector<std::unique_ptr<Node>>, 4> CreateGraph() {
  //                               -> var4 --
  //                             /            \/
  // -> var1 -> op1 -> var2 -> op2 -> var3 -> op3 -> var5
  //
  // cluster: [op1, op2, op3]
  // cluster_inputs: [var1]
  // cluster_outputs: [var5]
  // cluster_internals: [var2, var3, var4]

  // index : meaning
  // 0 : cluster
  // 1 : cluster_inputs
  // 2 : cluster_outputs
  // 3 : cluster_internals
  std::array<std::vector<std::unique_ptr<Node>>, 4> graph;

  // insert cluster_inputs var node
  framework::VarDesc var_desc_1("var1");
  var_desc_1.SetType(framework::proto::VarType::LOD_TENSOR);
  var_desc_1.SetDataType(framework::proto::VarType::FP32);
  var_desc_1.SetShape({64});
  graph.at(1).emplace_back(framework::ir::CreateNodeForTest(&var_desc_1));
  auto* var1 = graph.at(1).back().get();

  // insert cluster_internals var node
  framework::VarDesc var_desc_2("var2");
  var_desc_2.SetType(framework::proto::VarType::LOD_TENSOR);
  var_desc_2.SetDataType(framework::proto::VarType::INT32);
  var_desc_2.SetShape({64, 128});
  graph.at(2).emplace_back(framework::ir::CreateNodeForTest(&var_desc_2));
  auto* var2 = graph.at(2).back().get();

  framework::VarDesc var_desc_3("var3");
  var_desc_3.SetType(framework::proto::VarType::LOD_TENSOR);
  var_desc_3.SetDataType(framework::proto::VarType::INT32);
  var_desc_3.SetShape({64, 128, 1024});
  graph.at(2).emplace_back(framework::ir::CreateNodeForTest(&var_desc_3));
  auto* var3 = graph.at(2).back().get();

  framework::VarDesc var_desc_4("var4");
  var_desc_4.SetType(framework::proto::VarType::LOD_TENSOR);
  var_desc_4.SetDataType(framework::proto::VarType::INT32);
  var_desc_4.SetShape({64, 128, 1024, 4096});
  graph.at(2).emplace_back(framework::ir::CreateNodeForTest(&var_desc_4));
  auto* var4 = graph.at(2).back().get();

  // insert cluster_outputs var node
  framework::VarDesc var_desc_5("var5");
  var_desc_5.SetType(framework::proto::VarType::LOD_TENSOR);
  var_desc_5.SetDataType(framework::proto::VarType::BOOL);
  var_desc_5.SetShape({64, 128, 1024, 4096, 32});
  graph.at(3).emplace_back(framework::ir::CreateNodeForTest(&var_desc_5));
  auto* var5 = graph.at(3).back().get();

  // insert cluster op node
  framework::OpDesc op_desc_1("op1", {{"var1", {""}}}, {{"var2", {""}}}, {});
  graph.at(0).emplace_back(framework::ir::CreateNodeForTest(&op_desc_1));
  auto* op1 = graph.at(0).back().get();

  op1->inputs.emplace_back(var1);
  var1->outputs.emplace_back(op1);

  op1->outputs.emplace_back(var2);
  var2->inputs.emplace_back(op1);

  framework::OpDesc op_desc_3("op3", {{"var3", {""}}, {"var4", {""}}},
                              {{"var5", {""}}}, {});
  graph.at(0).emplace_back(framework::ir::CreateNodeForTest(&op_desc_3));
  auto* op3 = graph.at(0).back().get();

  op3->inputs.emplace_back(var3);
  var3->outputs.emplace_back(op3);

  op3->inputs.emplace_back(var4);
  var4->outputs.emplace_back(op3);

  op3->outputs.emplace_back(var5);
  var5->inputs.emplace_back(op3);

  framework::OpDesc op_desc_2("op2", {{"var2", {""}}},
                              {{"var3", {""}}, {"var4", {""}}}, {});
  graph.at(0).emplace_back(framework::ir::CreateNodeForTest(&op_desc_2));
  auto* op2 = graph.at(0).back().get();

  op2->inputs.emplace_back(var2);
  var2->outputs.emplace_back(op2);

  op2->outputs.emplace_back(var3);
  var3->inputs.emplace_back(op2);

  op2->outputs.emplace_back(var4);
  var4->inputs.emplace_back(op2);

  return graph;
}

void CreateCluster(GraphNodeVec* cluster, GraphNodeVec* cluster_inputs,
                   GraphNodeVec* cluster_outputs,
                   GraphNodeVec* cluster_internals) {
  // static to avoid destroy
  static const auto& graph = CreateGraph();

  // generate cluster
  for (const auto& node : graph.at(0)) {
    cluster->emplace_back(node.get());
  }
  // generate cluster_inputs
  for (const auto& node : graph.at(1)) {
    cluster_inputs->emplace_back(node.get());
  }
  // generate cluster_internals
  for (const auto& node : graph.at(2)) {
    cluster_internals->emplace_back(node.get());
  }
  // generate cluster_outputs
  for (const auto& node : graph.at(3)) {
    cluster_outputs->emplace_back(node.get());
  }
}

}  // namespace piano
}  // namespace paddle

#define REGISTER_TEST_OP(name)                              \
  REGISTER_OP_WITHOUT_GRADIENT(name, paddle::piano::TestOp, \
                               paddle::piano::TestOpMaker); \
  REGISTER_PIANO_OP(name, paddle::piano::TestPianoOpMaker,  \
                    paddle::piano::TestPianoOpKernel)

REGISTER_TEST_OP(op1)
REGISTER_TEST_OP(op2)
REGISTER_TEST_OP(op3)

#undef REGISTER_TEST_OP

namespace paddle {
namespace piano {

TEST(GraphExecutorTest, basic) {
  GraphNodeVec cluster, cluster_inputs, cluster_outputs, cluster_internals;
  CreateCluster(&cluster, &cluster_inputs, &cluster_outputs,
                &cluster_internals);

  PianoGraphExecutor exec(100, cluster, cluster_inputs, cluster_outputs,
                          cluster_internals);
  auto module_proto = exec();

  // check topologic sorting right
  ASSERT_EQ(GetCompileOrder(), std::vector<std::string>({"op1", "op2", "op3"}));

  // check module proto right
  ASSERT_TRUE(module_proto.has_name());
  ASSERT_TRUE(module_proto.has_entry_function_signature());

  const auto& entry_sig = module_proto.entry_function_signature();
  ASSERT_EQ(entry_sig.parameters_size(), cluster_inputs.size());
  for (int i = 0; i < entry_sig.parameter_names_size(); ++i) {
    ASSERT_NE(entry_sig.parameter_names(i).find(cluster_inputs[i]->Name()),
              std::string::npos);
  }

  const auto& entry_param = entry_sig.parameters(0);
  ASSERT_TRUE(entry_param.has_element_type());
  ASSERT_EQ(entry_param.element_type(),
            utils::VarType2NoteType(framework::proto::VarType::FP32));
  ASSERT_EQ(entry_param.dimensions_size(), 1);
  ASSERT_EQ(entry_param.dimensions(0), 64);
}

}  // namespace piano
}  // namespace paddle
