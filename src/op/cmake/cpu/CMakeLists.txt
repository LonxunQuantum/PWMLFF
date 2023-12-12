# 4. add_library
add_library(CalcOps_cpu SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_cpu.cpp)
target_include_directories(CalcOps_cpu PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(CalcOps_cpu PRIVATE ${TORCH_LIBRARIES})
# target_link_libraries(CalcOps_cpu PRIVATE "/data/home/hfhuang/software/anaconda3/envs/pytorch2/lib/python3.11/site-packages/torch/lib/libtorch.so")
# target_link_libraries(CalcOps_cpu PRIVATE "/data/home/hfhuang/software/anaconda3/envs/pytorch2/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so")
target_link_libraries(CalcOps_cpu PRIVATE ${TORCH_PATH}/lib/libtorch.so)
target_link_libraries(CalcOps_cpu PRIVATE ${TORCH_PATH}/lib/libtorch_cpu.so)

add_library(CalcOps_bind_cpu SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_bind_cpu.cpp)
target_compile_features(CalcOps_bind_cpu PRIVATE cxx_std_17)
target_link_libraries(CalcOps_bind_cpu PRIVATE CalcOps_cpu ${TORCH_LIBRARIES})
