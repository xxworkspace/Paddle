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
#pragma once

#include <memory>
#include <string>
#include <vector>

namespace paddle {
namespace piano {

namespace note {
class Module;
}

namespace backends {

class ScheduleWrapper {
 public:
 protected:
  std::string kernel_name_;
};
using Schedules = std::vector<std::unique_ptr<ScheduleWrapper>>;

// Compiler is an abstract class for compilation on a particular platform
//
// 'Compiler' ties together note::instruction and codegen (CG) to generate
// efficient binary code for the target platform.
//
// 'XXCompiler' class for a particular device inherit 'Compiler' and
// overwrite the function 'Apply'

class Compiler {
 public:
  Compiler() = default;
  virtual ~Compiler() {}

  virtual Schedules Apply(std::unique_ptr<note::Module>&) = 0;
};

}  // namespace backends
}  // namespace piano
}  // namespace paddle
