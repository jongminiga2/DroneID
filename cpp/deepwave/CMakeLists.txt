cmake_minimum_required(VERSION 3.10)
project(remove_turbo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(remove_turbo remove_turbo.cpp)

target_compile_options(remove_turbo PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -Wall -Wextra>
    $<$<CXX_COMPILER_ID:MSVC>:/O2 /W3>
)

# ARM cross-compile example (pass via cmake -DCMAKE_TOOLCHAIN_FILE=...):
#   cmake -DCMAKE_TOOLCHAIN_FILE=arm-toolchain.cmake ..
#
# Minimal arm-toolchain.cmake:
#   set(CMAKE_SYSTEM_NAME Linux)
#   set(CMAKE_SYSTEM_PROCESSOR arm)
#   set(CMAKE_C_COMPILER   arm-linux-gnueabihf-gcc)
#   set(CMAKE_CXX_COMPILER arm-linux-gnueabihf-g++)
