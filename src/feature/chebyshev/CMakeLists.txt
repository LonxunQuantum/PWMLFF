cmake_minimum_required(VERSION 3.27)
project(descriptor_pybind)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)

execute_process(COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)" OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH) ### cmake
string(REPLACE "\n" "" TORCH_CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}) 
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH}) 
list(APPEND CMAKE_CXX_FLAGS ${TORCH_CXX_FLAGS})
execute_process(COMMAND python3 -c "import pybind11; print(pybind11.__path__[0])" OUTPUT_VARIABLE PYBIND11_PATH)
string(REPLACE "\n" "/share/cmake" PYBIND11_CMAKE_PREFIX_PATH "${PYBIND11_PATH}")
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_CMAKE_PREFIX_PATH})
find_package(pybind11 CONFIG REQUIRED)



set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
file(MAKE_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
file(MAKE_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/obj)
file(MAKE_DIRECTORY ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})


# add_executable(test_basic ${CMAKE_CURRENT_SOURCE_DIR}/test/test_basic.cpp)
# add_executable(test_radial ${CMAKE_CURRENT_SOURCE_DIR}/test/test_radial.cpp)
# add_executable(neighborList ${CMAKE_CURRENT_SOURCE_DIR}/src/neighborList.cpp)
# add_executable(descriptor ${CMAKE_CURRENT_SOURCE_DIR}/src/descriptor.cpp)
add_library(neighborList SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/neighborList.cpp)
add_library(descriptor SHARED ${CMAKE_CURRENT_SOURCE_DIR}/src/descriptor.cpp)
# add_library(test_radial SHARED ${CMAKE_CURRENT_SOURCE_DIR}/test/test_radial.cpp)
# target_include_directories(test_radial PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_include_directories(test_basic PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# target_include_directories(neighborList PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(neighborList PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(descriptor PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(descriptor PRIVATE neighborList)
# target_compile_features(test_radial PUBLIC cxx_std_17)
# target_compile_features(test_basic PUBLIC cxx_std_17)


# Add descriptor_pybind module
pybind11_add_module(descriptor_pybind ${CMAKE_CURRENT_SOURCE_DIR}/src/descriptor_pybind.cpp)

# Link the libraries
target_link_libraries(descriptor_pybind PRIVATE descriptor)