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

#include <cuda.h>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "paddle/fluid/compiler/piano/backends/schedule_wrapper.h"

namespace piano {
namespace gpu {

class NvptxCustomParams {
public:
    Dim3 dg_;
    Dim3 db_;
    uint32_t num_vals_;
    DataType data_type_;

    std::vector<std::string> input_names;
    std::vector<std::string> input_names;
};

class NvptxCustomSchedule : public ScheduleWrapper {
public:
    NvptxCustomSchedule(){}
    ~NvptxCustomSchedule(){}

    virtual Status Run(const ExecutionContext&) = 0;
private:
    CUfunction cu_function_;
    NvptxCustomParams nvptx_custom_params_;
};

}
}