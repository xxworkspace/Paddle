/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <fstream>
#include <iostream>
#include "paddle/fluid/inference/tests/api/tester_helper.h"

DEFINE_string(infer_label, "", "label file");

namespace paddle {
namespace inference {
namespace analysis {

void SetConfig(AnalysisConfig *cfg) {
  cfg->SetModel(FLAGS_infer_model + "/model", FLAGS_infer_model + "/params");
  // cfg->DisableGpu();
  cfg->EnableUseGpu(200, 0);
  cfg->SwitchIrOptim(true);
  cfg->SwitchSpecifyInputNames();
  cfg->SetCpuMathLibraryNumThreads(FLAGS_paddle_num_threads);
  cfg->pass_builder()->DeletePass("identity_scale_op_clean_pass");
}

void PrepareInputs(std::vector<PaddleTensor> *input_slots, int batch_size) {
  std::vector<std::vector<float>> data;
  std::vector<int64_t> data_dims{batch_size, 3, 1024, 2048};
  GetInput<float>(FLAGS_infer_data, &data, data_dims);
  std::vector<std::vector<int32_t>> label;
  std::vector<int64_t> label_dims{batch_size, 1024, 2048};
  GetInput<int32_t>(FLAGS_infer_label, &label, label_dims);
  PaddleTensor data_tensor;
  data_tensor.name = "img";
  data_tensor.shape.assign({batch_size, 3, 1024, 2048});
  data_tensor.dtype = PaddleDType::FLOAT32;
  TensorAssignData<float>(&data_tensor, data);
  PaddleTensor label_tensor;
  label_tensor.name = "label";
  label_tensor.shape.assign({batch_size, 1024, 2048});
  label_tensor.dtype = PaddleDType::INT32;
  TensorAssignData<int32_t>(&label_tensor, label);
  input_slots->push_back(std::move(data_tensor));
  input_slots->push_back(std::move(label_tensor));
}

void SetInput(std::vector<std::vector<PaddleTensor>> *inputs) {
  std::vector<PaddleTensor> input_slots;
  int test_batch_num = 1;
  LOG(INFO) << "The number of samples to be test: "
            << test_batch_num * FLAGS_batch_size;
  for (int bid = 0; bid < test_batch_num; ++bid) {
    input_slots.clear();
    PrepareInputs(&input_slots, FLAGS_batch_size);
    (*inputs).emplace_back(std::move(input_slots));
  }
}

// Easy for profiling independently.
void profile(bool use_mkldnn = false) {
  AnalysisConfig cfg;
  SetConfig(&cfg);

  if (use_mkldnn) {
    cfg.EnableMKLDNN();
  }
  std::vector<PaddleTensor> outputs;

  std::vector<std::vector<PaddleTensor>> input_slots_all;
  SetInput(&input_slots_all);
  TestPrediction(reinterpret_cast<const PaddlePredictor::Config *>(&cfg),
                 input_slots_all, &outputs, FLAGS_num_threads);
  size_t size = GetSize(outputs[0]);
  float *result = static_cast<float *>(outputs[0].data.data());
  for (size_t i = 0; i < size; i++) {
    std::cout << "*******Result*******: " << result[i] << std::endl;
  }
}

TEST(Analyzer_deeplabv3p, profile) { profile(); }
#ifdef PADDLE_WITH_MKLDNN
TEST(Analyzer_deeplabv3p, profile_mkldnn) { profile(true /* use_mkldnn */); }
#endif

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
