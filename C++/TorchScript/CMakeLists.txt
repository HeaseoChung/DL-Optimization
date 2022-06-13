cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)
set(CMAKE_CXX_COMPILER g++)
list(APPEND CMAKE_PREFIX_PATH "/workspace/ffmpeg/ffmpeg_sources/libtorch-1.8.0/", "/workspace/opencv/build")

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(libtorch-build test.cpp)
target_link_libraries(libtorch-build ${TORCH_LIBRARIES})
target_link_libraries(libtorch-build ${OpenCV_LIBRARIES})

set_property(TARGET libtorch-build PROPERTY CXX_STANDARD 14)
