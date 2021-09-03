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

#include "paddle/fluid/compiler/paddle2piano/vartype_utils.h"

#include <unordered_map>
#include <unordered_set>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace piano {
namespace utils {

using framework::proto::VarType;

namespace {
const std::unordered_map<framework::proto::VarType::Type,
                         note::ElementTypeProto>&
GetVarType2NoteTypeMap() {
  static std::unordered_map<framework::proto::VarType::Type,
                            note::ElementTypeProto>
      vartype2notetype = {{framework::proto::VarType::BOOL, note::B1},
                          {framework::proto::VarType::INT8, note::S8},
                          {framework::proto::VarType::INT16, note::S16},
                          {framework::proto::VarType::INT32, note::S32},
                          {framework::proto::VarType::INT64, note::S64},
                          {framework::proto::VarType::FP16, note::F16},
                          {framework::proto::VarType::FP32, note::F32},
                          {framework::proto::VarType::FP64, note::F64},
                          {framework::proto::VarType::UINT8, note::U8},
                          {framework::proto::VarType::SIZE_T, note::U64}};
  return vartype2notetype;
}
}  // namespace

note::ElementTypeProto VarType2NoteType(framework::proto::VarType::Type type) {
  const auto& vartype2notetype = GetVarType2NoteTypeMap();
  PADDLE_ENFORCE_NE(vartype2notetype.find(type), vartype2notetype.end(),
                    "Unsupported value data type (%s)",
                    framework::DataTypeToString(type).c_str());
  return vartype2notetype.at(type);
}

VarType::Type GetVarDataType(const framework::VarDesc* var) {
  auto var_type = var->GetType();

  // non-pod type list
  static std::unordered_set<VarType::Type> non_pod_types = {
      VarType::LOD_TENSOR, VarType::SELECTED_ROWS, VarType::LOD_TENSOR_ARRAY};
  if (non_pod_types.count(var_type) != 0) {
    // if the value type is non-pod type
    var_type = var->GetDataType();
  }

  // check whether value type is supported
  if (GetVarType2NoteTypeMap().count(var_type) != 0) {
    // if value type is supported type
    return var_type;
  }

  PADDLE_THROW(platform::errors::Unimplemented(
      "Unsupported value data type (%s)",
      framework::DataTypeToString(var_type).c_str()));
  return framework::proto::VarType::RAW;
}

}  // namespace utils
}  // namespace piano
}  // namespace paddle
