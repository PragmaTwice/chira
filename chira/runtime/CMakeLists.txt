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

add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/chirart.ll.inc
        ${CMAKE_CURRENT_BINARY_DIR}/chirart.ll
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/generate-ir.sh chirart.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/generate-ir.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/chirart.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/chirart.h
)

add_custom_command(
    OUTPUT
        ${CMAKE_CURRENT_BINARY_DIR}/chirart_main.ll.inc
        ${CMAKE_CURRENT_BINARY_DIR}/chirart_main.ll
    COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/generate-ir.sh chirart_main.cpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/generate-ir.sh
        ${CMAKE_CURRENT_SOURCE_DIR}/chirart_main.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/chirart.h
)

add_custom_target(chirart-generate-ir
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/chirart.ll.inc
        ${CMAKE_CURRENT_BINARY_DIR}/chirart.ll
)

add_custom_target(chirart-generate-ir-main
    DEPENDS
        ${CMAKE_CURRENT_BINARY_DIR}/chirart_main.ll.inc
        ${CMAKE_CURRENT_BINARY_DIR}/chirart_main.ll
)

add_library(ChiraRuntime
    Runtime.h
    Runtime.cpp
)

add_dependencies(ChiraRuntime
    chirart-generate-ir
    chirart-generate-ir-main
)
