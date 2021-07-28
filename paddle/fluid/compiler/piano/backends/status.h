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

#include <string>

namespace piano {

enum ErrorCode {
    OK = 0,
};

class Status {
public:
    Status(ErrorCode error_code, std::string msg)
        :status_(status), msg_(msg){}
    ~Status()

    bool OK();
    ErrorCode Code();
    std::string Msg();
private:
    std::string msg_;
    ErrorCode error_code_;
}

}