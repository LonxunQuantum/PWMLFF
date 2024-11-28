# include 
find_package(Torch REQUIRED)
# add_library
add_library(CalcOps_cuda SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps.cpp)
target_include_directories(CalcOps_cuda PRIVATE ${CMAKE_SOURCE_DIR})
target_link_libraries(CalcOps_cuda PRIVATE ${TORCH_LIBRARIES})

# register_op
file(GLOB CPP_FILES ${CMAKE_SOURCE_DIR}/src/*.cpp)
file(GLOB CU_FILES ${CMAKE_SOURCE_DIR}/kernel/*.cu)
file(GLOB CU_FILES_UTILITIES ${CMAKE_SOURCE_DIR}/kernel/utilities/*.cu)
list(APPEND CU_FILES ${CU_FILES_UTILITIES})
file(GLOB CALC_FILES ${CMAKE_SOURCE_DIR}/src/CalcOp*.cpp)
list(REMOVE_ITEM CPP_FILES ${CALC_FILES})

foreach(CPP_FILE ${CPP_FILES})
	get_filename_component(CPP_LIB ${CPP_FILE} NAME)
	add_library(${CPP_LIB} SHARED ${CPP_FILE})
	target_link_libraries(CalcOps_cuda PRIVATE ${CPP_LIB} ${TORCH_LIBRARIES})
endforeach(CPP_FILE)

foreach(CU_FILE ${CU_FILES})
	get_filename_component(CU_LIB ${CU_FILE} NAME)
	add_library(${CU_LIB} SHARED ${CU_FILE})
	target_link_libraries(CalcOps_cuda PRIVATE ${CU_LIB} ${TORCH_LIBRARIES})
endforeach(CU_FILE)

add_library(CalcOps_bind SHARED ${CMAKE_SOURCE_DIR}/src/CalcOps_bind.cpp)
target_compile_features(CalcOps_bind PRIVATE cxx_std_17)
target_link_libraries(CalcOps_bind PRIVATE CalcOps_cuda ${TORCH_LIBRARIES})