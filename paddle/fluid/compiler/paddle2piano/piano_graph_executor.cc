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

#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/compiler/paddle2piano/piano_op_kernel_context.h"
#include "paddle/fluid/compiler/paddle2piano/vartype_utils.h"
#include "paddle/fluid/compiler/piano/symbolization/meta_op.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {

using framework::ir::Node;
using GraphNodeVec = PianoGraphExecutor::GraphNodeVec;

std::unique_ptr<PianoScope> PianoGraphExecutor::CreateInputOperand(
    symbolization::NoteBuilder* builder) const {
  std::unique_ptr<PianoScope> scope = std::make_unique<PianoScope>();
  for (int64_t id = 0; id < cluster_inputs_.size(); ++id) {
    auto* node = cluster_inputs_.at(id);
    PADDLE_ENFORCE_EQ(node->IsVar(), true,
                      platform::errors::InvalidArgument(
                          "Cluster Sub-Graph Input should be var"));

    const auto& var_name = node->Name();

    // create operand shape
    const auto& var_shape = node->Var()->GetShape();
    const auto& var_type = utils::GetVarDataType(node->Var());

    // convert framework vartype to piano note type
    note::ElementTypeProto element_type = utils::VarType2NoteType(var_type);
    Shape operand_shape(element_type, var_shape);

    // create Operand
    symbolization::Operand op =
        symbolization::Parameter(builder, id, operand_shape, var_name);

    // store into PianoScope
    scope->SetOperand(var_name, op);
  }
  return scope;
}

GraphNodeVec PianoGraphExecutor::SortInternalCluster() const {
  GraphNodeVec cluster_sorted;
  std::unordered_set<Node*> cluster_set(cluster_.cbegin(), cluster_.cend());

  std::unordered_map<Node*, size_t> indegree;
  std::unordered_map<Node*, std::unordered_map<Node*, size_t>> adj_list;
  std::queue<Node*> topo_queue;

  // record all op's input op and output op
  for (auto* n : cluster_) {
    PADDLE_ENFORCE_EQ(n->IsOp(), true,
                      platform::errors::PreconditionNotMet(
                          "Cluster's node all should be op node"));
    PADDLE_ENFORCE_EQ(PianoOpRegistry::IsPianoOp(n->Name()), true,
                      platform::errors::PreconditionNotMet(
                          "Cluster's op all should be piano op"));
    // the op's input is var
    for (auto* in_var : n->inputs) {
      // the var's input is op
      for (auto* in_op : in_var->inputs) {
        if (cluster_set.find(in_op) != cluster_set.end()) {
          ++indegree[n];
          ++adj_list[in_op][n];
        }
      }
    }
  }

  // find topology entrance
  for (auto* n : cluster_) {
    if (indegree[n] == 0) {
      topo_queue.push(n);
    }
  }

  // topological sorting
  while (!topo_queue.empty()) {
    auto* cur_op = topo_queue.front();
    topo_queue.pop();

    cluster_sorted.emplace_back(cur_op);
    for (const auto& adj_pair : adj_list[cur_op]) {
      // decrease output op's in-degree
      indegree.at(adj_pair.first) -= adj_pair.second;

      // if empty, push into queue
      if (indegree.at(adj_pair.first) == 0) {
        topo_queue.push(adj_pair.first);
      }
    }
  }

  PADDLE_ENFORCE_EQ(cluster_sorted.size(), cluster_.size(),
                    platform::errors::PreconditionNotMet(
                        "Cluster Sub-Graph shouldn't contain cycle."));
  return cluster_sorted;
}

void PianoGraphExecutor::RunCompile(const GraphNodeVec& cluster,
                                    PianoScope* scope,
                                    symbolization::NoteBuilder* builder) const {
  for (auto* n : cluster) {
    const auto& op_name = n->Name();
    const auto* op_desc = n->Op();

    const auto& op_kernel_map = PianoOpRegistry::AllPianoOpKernels(op_name);
    // TODO(jiangcheng05): how to distinguish library's kernel, like cudnn?
    op_kernel_map.at("PLAIN")(PianoOpKernelContext(op_desc, scope, builder));
  }
}

note::ModuleProto PianoGraphExecutor::operator()() const {
  // Step1: create unique NoteBuilder
  std::string builder_name = "NoteBuilderOfGraph_";
  builder_name.append(std::to_string(graph_id_));

  symbolization::NoteBuilder builder(builder_name);

  // Step2: create graph's input operand
  auto scope = CreateInputOperand(&builder);

  // Step3: topo sort graph
  // rvalue references avoid useless copy
  const auto& cluster_sorted = SortInternalCluster();

  // Step4: get PianoOpKernel and run compile
  RunCompile(cluster_sorted, scope.get(), &builder);

  // Step5: build and return module
  return builder.Build();
}

}  // namespace piano
}  // namespace paddle
