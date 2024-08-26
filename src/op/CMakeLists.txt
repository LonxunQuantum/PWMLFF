#add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# 1. Basic info 
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
project(CalcOps)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
# enable_language(CUDA)			### no need, just compile on GPU device
# set(CMAKE_CUDA_FLAGS "-arch=sm_60")

# 2. include 
execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH) ### cmake
string(REPLACE "\n" "" TORCH_CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}) 
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}) 
# find_package(Torch REQUIRED)		# if use find_package, then it will link all library in torch, including torch_cpu, torch_cuda...
list(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})


# 3. set 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})

# 4. add_library
execute_process(COMMAND python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])" OUTPUT_VARIABLE PYTHON_INCLUDE_DIR) ### python
string(REPLACE "\n" "" PYTHON_INCLUDE_DIR ${PYTHON_INCLUDE_DIR}) ### python
execute_process(COMMAND python3 -c "import torch; print(torch.__path__[0])" OUTPUT_VARIABLE TORCH_PATH) ### torch
string(REPLACE "\n" "" TORCH_PATH ${TORCH_PATH}) ### torch
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${TORCH_PATH}/include/)
include_directories(${TORCH_PATH}/include/torch/csrc/api/include/)

add_subdirectory(cmake/gpu)
add_subdirectory(cmake/cpu)
