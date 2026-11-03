cmake_minimum_required(VERSION 3.24)
project(bdcudatag VERSION 0.1.0 LANGUAGES CXX CUDA)

option(BDCUDATAG_BUILD_TESTS "Build the detector tests" ON)
option(BDCUDATAG_BUILD_BENCHMARKS "Build the benchmark executable" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Orin Nano Super (Ampere, SM 8.7) is the target.  Override for other GPUs:
#   -DCMAKE_CUDA_ARCHITECTURES=86   (desktop Ampere, used for local testing)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 87)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Threads REQUIRED)

add_library(bdcudatag SHARED
    src/detector.cpp
    src/gpu_detector.cu
)
add_library(bdcudatag::bdcudatag ALIAS bdcudatag)

target_compile_definitions(bdcudatag PRIVATE BDCUDATAG_BUILDING)
target_include_directories(bdcudatag
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(bdcudatag PRIVATE CUDA::cudart)

set_target_properties(bdcudatag PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

if(MSVC)
    target_compile_options(bdcudatag PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(bdcudatag PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>)
endif()

include(CTest)
if(BDCUDATAG_BUILD_TESTS AND BUILD_TESTING)
    add_executable(bdcudatag_tests tests/test_detector.cpp)
    target_link_libraries(bdcudatag_tests PRIVATE bdcudatag CUDA::cudart Threads::Threads)
    target_include_directories(bdcudatag_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    add_test(NAME bdcudatag_tests COMMAND bdcudatag_tests)
endif()

if(BDCUDATAG_BUILD_BENCHMARKS)
    add_executable(bdcudatag_bench bench/bench_detector.cpp)
    target_link_libraries(bdcudatag_bench PRIVATE bdcudatag)
    target_include_directories(bdcudatag_bench PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)

    # Exports scenes for compare_detectors.py.  No detector or CUDA dependency:
    # it only renders tags from the family tables.
    add_executable(bdcudatag_scenes bench/scene_gen.cpp)
    target_include_directories(bdcudatag_scenes PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src ${CMAKE_CURRENT_SOURCE_DIR}/tests)
endif()

install(TARGETS bdcudatag
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
)
install(FILES include/bdcudatag/bdcudatag.h include/bdcudatag/detector.hpp
    DESTINATION include/bdcudatag)
