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
#include "llvm/Support/Host.h"
#include "paddle/fluid/compiler/piano/note/instruction.h"
#include "paddle/fluid/compiler/piano/note/opcode.h"
#include "paddle/fluid/compiler/piano/note_builder.h"

namespace paddle {
namespace piano {
namespace backends {

TEST(NvptxCompiler, Apply) {
  NoteBuilder note_builder("test_note_builder");

  std::vector<Operand> ops;
  ops.push_back(note_builder.AppendInstruction(note::InstructionProto(),
                                               note::OpCode::kConstant, {}));
  ops.push_back(note_builder.AppendInstruction(note::InstructionProto(),
                                               note::OpCode::kConstant, {}));
  note_builder.AppendInstruction(note::InstructionProto(), note::OpCode::kAdd,
                                 ops);

  auto note_proto = note_builder.Build();
  note::Module note_module(note_proto);

  NvptxCompiler nvptx_compiler;
  // note::Module note_module;
  auto kernel_executable_map = nvptx_compiler.Apply(&note_module);
  //
}

}  // namespace backends
}  // namespace piano
}  // namespace paddle
