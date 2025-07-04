#!/usr/bin/env bash

#  Copyright 2025 PragmaTwice
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

set -e
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
INPUT_FILE=$SCRIPT_DIR/$1
OUTPUT_FILE=./${1%.*}.ll
set -ex
clang -std=c++17 -nostdlib++ -O3 -Wall -S -emit-llvm -o $OUTPUT_FILE $INPUT_FILE
cat $OUTPUT_FILE | xxd -i - $OUTPUT_FILE.inc
