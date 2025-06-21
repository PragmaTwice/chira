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

usage() {
    echo "Usage: $0 <object file> <path of libc>"
    exit 1
}

OBJ_FILE=$1
if [ -z "$OBJ_FILE" ]; then
    usage
fi
if [[ ! "$OBJ_FILE" =~ \.o$ ]]; then
    echo "Error: $1 does not have a .o suffix"
    exit 1
fi

EXE_FILE=$(basename "$OBJ_FILE" .o)

LIBC_DIR=$2
if [ -z "$LIBC_DIR" ]; then
    usage
fi
if [ ! -d "$LIBC_DIR" ]; then
    echo "Error: $2 is not a valid directory"
    exit 1
fi

set -ex
ld $LIBC_DIR/crt1.o $LIBC_DIR/crti.o $OBJ_FILE $LIBC_DIR/libc.a $LIBC_DIR/crtn.o -o $EXE_FILE
