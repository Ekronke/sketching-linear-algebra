cmake_minimum_required(VERSION 3.18)
project(sketchla LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(_kernels cpp/kernels.cpp)
if(NOT MSVC)
  target_compile_options(_kernels PRIVATE -O3)
endif()

install(TARGETS _kernels DESTINATION sketchla)
