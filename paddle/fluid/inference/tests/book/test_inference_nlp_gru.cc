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

#include <sys/time.h>
#include <time.h>
#include <fstream>
#include <thread>  // NOLINT
#include "gflags/gflags.h"
#include "gtest/gtest.h"
#include "paddle/fluid/inference/tests/test_helper.h"
#include "paddle/fluid/platform/cpu_helper.h"

DEFINE_string(model_path, "", "Directory of the inference model.");
DEFINE_bool(prepare_vars, true, "Prepare variables before executor");

// This function just give dummy data for recognize_digits model.
size_t DummyData(std::vector<paddle::framework::LoDTensor>* out) {
  paddle::framework::LoDTensor words;
  std::vector<int64_t> ids{1064, 1603, 644, 699, 2878, 1219, 867,
                           1352, 8,    1,   13,  312,  479};
  paddle::framework::LoD lod{{0, ids.size()}};
  words.set_lod(lod);
  int64_t* pdata = words.mutable_data<int64_t>(
      {static_cast<int64_t>(ids.size()), 1}, paddle::platform::CPUPlace());
  memcpy(pdata, ids.data(), words.numel() * sizeof(int64_t));
  out->emplace_back(words);
  return 1;
}

// Load the input word index data from file and save into LodTensor.
// Return the size of words.
size_t LoadData(std::vector<paddle::framework::LoDTensor>* out) {
  return DummyData(out);
}

TEST(inference, nlp) {
  if (FLAGS_model_path.empty()) {
    LOG(FATAL) << "Usage: ./example --model_path=path/to/your/model";
  }

  std::vector<paddle::framework::LoDTensor> datasets;
  size_t num_total_words = LoadData(&datasets);
  LOG(INFO) << "Number of samples (seq_len<1024): " << datasets.size();
  LOG(INFO) << "Total number of words: " << num_total_words;

  // 0. Call `paddle::framework::InitDevices()` initialize all the devices
  std::unique_ptr<paddle::framework::Scope> scope(
      new paddle::framework::Scope());

  // 1. Define place, executor, scope
  paddle::platform::CPUPlace place;
  paddle::framework::Executor executor(place);

  // 2. Initialize the inference_program and load parameters
  std::unique_ptr<paddle::framework::ProgramDesc> inference_program;
  inference_program = InitProgram(&executor, scope.get(), FLAGS_model_path,
                                  /*model combined*/ false);
  // always prepare context
  std::unique_ptr<paddle::framework::ExecutorPrepareContext> ctx;
  ctx = executor.Prepare(*inference_program, 0);
  if (FLAGS_prepare_vars) {
    executor.CreateVariables(*inference_program, scope.get(), 0);
  }

  // preapre fetch
  const std::vector<std::string>& fetch_target_names =
      inference_program->GetFetchTargetNames();
  PADDLE_ENFORCE_EQ(fetch_target_names.size(), 1UL);
  std::map<std::string, paddle::framework::LoDTensor*> fetch_targets;
  paddle::framework::LoDTensor outtensor;
  fetch_targets[fetch_target_names[0]] = &outtensor;

  // prepare feed
  const std::vector<std::string>& feed_target_names =
      inference_program->GetFeedTargetNames();
  PADDLE_ENFORCE_EQ(feed_target_names.size(), 1UL);
  std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;

  // feed data and run
  for (size_t i = 0; i < datasets.size(); ++i) {
    feed_targets[feed_target_names[0]] = &(datasets[i]);
    executor.RunPreparedContext(ctx.get(), scope.get(), &feed_targets,
                                &fetch_targets, !FLAGS_prepare_vars);
  }
  // print the output
  for (auto iter = fetch_targets.begin(); iter != fetch_targets.end(); iter++) {
    LOG(INFO) << "Output name: " << iter->first;
    auto* out = iter->second;
    auto* out_data = out->data<int64_t>();
    for (int64_t i = 0; i < out->numel(); i++) {
      LOG(INFO) << out_data[i];
    }
  }
}
