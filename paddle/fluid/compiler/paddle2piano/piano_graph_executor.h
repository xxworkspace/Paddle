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

#pragma once

#include <vector>

#include "paddle/fluid/compiler/paddle2piano/piano_scope.h"
#include "paddle/fluid/compiler/piano/note/note.pb.h"
#include "paddle/fluid/compiler/piano/symbolization/note_builder.h"
#include "paddle/fluid/framework/ir/node.h"

namespace paddle {
namespace piano {

// An executor accept sub-graph which is generated by PianoCompilePass,
// run each op's PianoOpKernel, finally return the graph's ModuleProto.
//
// Parameter:
// 1. graph_id: the unique graph id, used for generating unique notebuilder name
// 2. cluster: a vector which contains all graph op, non-topological-sorting.
// 3. cluster_inputs: a vector which contains all graph's input var, the var's
//                    input are outside op, the output are inside op
// 4. cluster_outputs: a vector which contains all graph's output var, the var's
//                     input are inside op, the output are outside op
// 5. cluster_internals: a vector which contains all graph's internal var, the
//                        var's input and output are inside op
//
// Example:
//        -------------------------> op3 -> var4 ->
//      /                            /
// -> var1 -> op1 -> var2 -> op2 -> var3
//
// cluster: [op1, op2, op3]
// cluster_inputs: [var1]
// cluster_outputs: [var4]
// cluster_internals: [var2, var3]
//
// Describe:
// The executor consisted by the following step:
// 1. create a NoteBuilder, it's name is unique for each graph
// 2. create PianoScope, initially, scope only consist graph's input var and its
// operand
// 3. topological sorting graph
// 4. create PianoOpKernelContext and run each op's PianoOpKernel
// 5. run NoteBuilder's Build function to generate graph's ModuleProto
class PianoGraphExecutor {
 public:
  using GraphNodeVec = std::vector<framework::ir::Node*>;

  PianoGraphExecutor(int64_t graph_id, const GraphNodeVec& cluster,
                     const GraphNodeVec& cluster_inputs,
                     const GraphNodeVec& cluster_outputs,
                     const GraphNodeVec& cluster_internals)
      : graph_id_(graph_id),
        cluster_(cluster),
        cluster_inputs_(cluster_inputs),
        cluster_outputs_(cluster_outputs),
        cluster_internals_(cluster_internals) {}

  note::ModuleProto operator()() const;

 private:
  const int64_t graph_id_;
  const GraphNodeVec& cluster_;
  const GraphNodeVec& cluster_inputs_;
  const GraphNodeVec& cluster_outputs_;
  const GraphNodeVec& cluster_internals_;

  // create graph's input operand from cluster_inputs_
  // why return std::unique_ptr ? PianoScope DISABLE_COPY_AND_ASSIGN
  std::unique_ptr<PianoScope> CreateInputOperand(
      symbolization::NoteBuilder* builder) const;

  // run PianoOpKernel's Compile
  void RunCompile(const GraphNodeVec& cluster, PianoScope* scope,
                  symbolization::NoteBuilder* builder) const;

  // topologic sorting graph node
  GraphNodeVec SortInternalCluster() const;
};

}  // namespace piano
}  // namespace paddle