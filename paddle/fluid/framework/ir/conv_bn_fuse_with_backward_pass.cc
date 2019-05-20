// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/ir/conv_bn_fuse_with_backward_pass.h"
#include <functional>
#include <string>
#include <vector>
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/operators/math/cpu_vec.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace ir {

#define GET_CONV_BN_NODES(pattern_name)                                      \
  /* OPERATORS */                                                            \
  GET_IR_NODE_FROM_SUBGRAPH(conv, conv, pattern_name);                       \
  GET_IR_NODE_FROM_SUBGRAPH(batch_norm, batch_norm, pattern_name);           \
  /* CONV inputs */                                                          \
  GET_IR_NODE_FROM_SUBGRAPH(conv_weight, conv_weight, pattern_name);         \
  /* CONV outputs */                                                         \
  GET_IR_NODE_FROM_SUBGRAPH(conv_out, conv_out, pattern_name);               \
  /* BN inputs */                                                            \
  GET_IR_NODE_FROM_SUBGRAPH(bn_scale, bn_scale, pattern_name);               \
  GET_IR_NODE_FROM_SUBGRAPH(bn_bias, bn_bias, pattern_name);                 \
  GET_IR_NODE_FROM_SUBGRAPH(bn_mean, bn_mean, pattern_name);                 \
  GET_IR_NODE_FROM_SUBGRAPH(bn_variance, bn_variance, pattern_name);         \
  /* BN outputs */                                                           \
  GET_IR_NODE_FROM_SUBGRAPH(bn_out, bn_out, pattern_name); /* Out */         \
  GET_IR_NODE_FROM_SUBGRAPH(bn_mean_out, bn_mean_out, pattern_name);         \
  GET_IR_NODE_FROM_SUBGRAPH(bn_variance_out, bn_variance_out, pattern_name); \
  GET_IR_NODE_FROM_SUBGRAPH(bn_saved_mean, bn_saved_mean, pattern_name);     \
  GET_IR_NODE_FROM_SUBGRAPH(bn_saved_variance, bn_saved_variance, pattern_name)

static void recompute_bias_and_weights(
    LoDTensor* conv_weight_tensor, const LoDTensor& scale_tensor,
    const LoDTensor& bn_bias_tensor, const LoDTensor& mean_tensor,
    LoDTensor* variance_tensor, LoDTensor* eltwise_y_in_tensor, float epsilon) {
  using EigenVectorArrayMap =
      Eigen::Map<Eigen::Array<float, Eigen::Dynamic, 1>>;
  using ConstEigenVectorArrayMap =
      Eigen::Map<const Eigen::Array<float, Eigen::Dynamic, 1>>;
  using EigenMatrixArrayMap = Eigen::Map<
      Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

  // Re-compute bias of conv2d from BN
  PADDLE_ENFORCE_EQ(eltwise_y_in_tensor->dims(), bn_bias_tensor.dims());

  ConstEigenVectorArrayMap scale_array(scale_tensor.data<float>(),
                                       scale_tensor.numel(), 1);
  EigenVectorArrayMap variance_array(variance_tensor->data<float>(),
                                     variance_tensor->numel(), 1);
  ConstEigenVectorArrayMap mean_array(mean_tensor.data<float>(),
                                      mean_tensor.numel(), 1);
  ConstEigenVectorArrayMap bn_bias_array(bn_bias_tensor.data<float>(),
                                         bn_bias_tensor.numel(), 1);

  // variance will not be used anymore, so make it std_array and then tmp_array
  variance_array += epsilon;
  variance_array = variance_array.sqrt();
  variance_array = scale_array / variance_array;

  EigenVectorArrayMap eltwise_y_in_array(eltwise_y_in_tensor->data<float>(),
                                         eltwise_y_in_tensor->numel(), 1);

  eltwise_y_in_array =
      ((eltwise_y_in_array - mean_array) * variance_array) + bn_bias_array;

  // Re-compute weight of conv2d from BN
  auto conv_weight_shape = conv_weight_tensor->dims();
  auto conv_weight_shape_2d = flatten_to_2d(conv_weight_shape, 1);

  EigenMatrixArrayMap conv_weight_array_2d(conv_weight_tensor->data<float>(),
                                           conv_weight_shape_2d[0],
                                           conv_weight_shape_2d[1]);

  conv_weight_array_2d.colwise() *= variance_array;
}

static void to_cpu_tensor(const framework::LoDTensor& gpu_tensor,
                          framework::LoDTensor* cpu_tensor) {
  cpu_tensor->Resize(gpu_tensor.dims());
  cpu_tensor->mutable_data<float>(platform::CPUPlace());
  TensorCopySync(gpu_tensor, cpu_tensor->place(), cpu_tensor);
}

void ConvBNFuseWithBackwardPass::ApplyImpl(ir::Graph* graph) const {
  PADDLE_ENFORCE(graph);
  FusePassBase::Init(name_scope_, graph);

  auto* scope = param_scope();
  GraphPatternDetector gpd;
  auto* conv_input =
      gpd.mutable_pattern()
          ->NewNode(patterns::PDNodeName(name_scope_, "conv_input"))
          ->AsInput()
          ->assert_is_op_input("conv2d", "Input");
  patterns::ConvBN conv_bn_pattern(gpd.mutable_pattern(), name_scope_);
  conv_bn_pattern(conv_input, false /*with_eltwise_add*/);

  auto handler = [&](const GraphPatternDetector::subgraph_t& subgraph,
                     Graph* g) {
    VLOG(4) << "handle ConvBN fuse forward";

    // conv, batch_norm,
    // conv_weight, conv_out,
    // bn_scale, bn_bias, bn_mean, bn_variance,
    // bn_out, bn_mean_out, bn_variance_out, bn_saved_mean,
    // bn_saved_variance
    GET_CONV_BN_NODES(conv_bn_pattern);

    // Get batch norm bias
    auto* bn_bias_tensor =
        scope->FindVar(bn_bias->Name())->GetMutable<LoDTensor>();

    // Create eltwise_y (conv bias) variable
    VarDesc eltwise_y_in_desc(
        patterns::PDNodeName(name_scope_, "eltwise_y_in"));
    eltwise_y_in_desc.SetShape(framework::vectorize(bn_bias_tensor->dims()));
    eltwise_y_in_desc.SetDataType(bn_bias_tensor->type());
    eltwise_y_in_desc.SetLoDLevel(bn_bias->Var()->GetLoDLevel());
    eltwise_y_in_desc.SetPersistable(true);
    auto* eltwise_y_in_node = g->CreateVarNode(&eltwise_y_in_desc);

    if (!(g->Has("__recompute_bias_and_weights__") &&
          !g->Get<bool>("__recompute_bias_and_weights__"))) {
      // for bn
      auto* scale_tensor =
          scope->FindVar(bn_scale->Name())->GetMutable<LoDTensor>();
      auto* variance_tensor =
          scope->FindVar(bn_variance->Name())->GetMutable<LoDTensor>();
      auto* mean_tensor =
          scope->FindVar(bn_mean->Name())->GetMutable<LoDTensor>();
      // for conv
      auto* conv_weight_tensor =
          scope->FindVar(conv_weight->Name())->GetMutable<LoDTensor>();
      // for elementwise_add bias
      auto* eltwise_y_in_tensor =
          scope->Var(eltwise_y_in_node->Name())->GetMutable<LoDTensor>();
      eltwise_y_in_tensor->Resize(bn_bias_tensor->dims());
      float epsilon = boost::get<float>(batch_norm->Op()->GetAttr("epsilon"));

      if (platform::is_gpu_place(bn_bias_tensor->place())) {
        // for bn
        framework::LoDTensor cpu_bn_bias_tensor;
        to_cpu_tensor(*bn_bias_tensor, &cpu_bn_bias_tensor);
        framework::LoDTensor cpu_scale_tensor;
        to_cpu_tensor(*scale_tensor, &cpu_scale_tensor);
        framework::LoDTensor cpu_variance_tensor;
        to_cpu_tensor(*variance_tensor, &cpu_variance_tensor);
        framework::LoDTensor cpu_mean_tensor;
        to_cpu_tensor(*mean_tensor, &cpu_mean_tensor);
        // for conv
        framework::LoDTensor cpu_conv_weight_tensor;
        to_cpu_tensor(*conv_weight_tensor, &cpu_conv_weight_tensor);
        // for elementwise_add bias
        eltwise_y_in_tensor->mutable_data<float>(bn_bias_tensor->place());
        framework::LoDTensor cpu_eltwise_y_in_tensor;
        cpu_eltwise_y_in_tensor.Resize(eltwise_y_in_tensor->dims());
        // Initialize eltwise_y
        std::fill_n(
            cpu_eltwise_y_in_tensor.mutable_data<float>(platform::CPUPlace()),
            cpu_eltwise_y_in_tensor.numel(), 0.0f);
        recompute_bias_and_weights(&cpu_conv_weight_tensor, cpu_scale_tensor,
                                   cpu_bn_bias_tensor, cpu_mean_tensor,
                                   &cpu_variance_tensor,
                                   &cpu_eltwise_y_in_tensor, epsilon);
        framework::TensorCopySync(cpu_eltwise_y_in_tensor,
                                  eltwise_y_in_tensor->place(),
                                  eltwise_y_in_tensor);
        framework::TensorCopySync(cpu_conv_weight_tensor,
                                  conv_weight_tensor->place(),
                                  conv_weight_tensor);
      } else {
        // for elementwise_add bias
        // Initialize eltwise_y
        std::fill_n(
            eltwise_y_in_tensor->mutable_data<float>(platform::CPUPlace()),
            eltwise_y_in_tensor->numel(), 0.0f);
        recompute_bias_and_weights(
            conv_weight_tensor, *scale_tensor, *bn_bias_tensor, *mean_tensor,
            variance_tensor, eltwise_y_in_tensor, epsilon);
      }
    }
    // fuse conv+bn into conv+elementwise_add
    // create an elementwise add node.
    OpDesc desc;
    desc.SetInput("X", std::vector<std::string>({conv_out->Name()}));
    desc.SetInput("Y", std::vector<std::string>({eltwise_y_in_node->Name()}));
    desc.SetOutput("Out", std::vector<std::string>({bn_out->Name()}));
    desc.SetType("elementwise_add");
    desc.SetAttr("axis", 1);
    desc.SetAttr("op_role", static_cast<int>(framework::OpRole::kForward));
    auto eltwise_op = g->CreateOpNode(&desc);  // OpDesc will be copied.

    GraphSafeRemoveNodes(graph, {bn_scale, bn_bias, bn_mean, bn_variance,
                                 batch_norm, bn_mean_out, bn_variance_out,
                                 bn_saved_mean, bn_saved_variance});

    IR_NODE_LINK_TO(conv_out, eltwise_op);
    IR_NODE_LINK_TO(eltwise_y_in_node, eltwise_op);
    IR_NODE_LINK_TO(eltwise_op, bn_out);
  };

  gpd(graph, handler);
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(conv_bn_fuse_with_backward_pass,
              paddle::framework::ir::ConvBNFuseWithBackwardPass);
