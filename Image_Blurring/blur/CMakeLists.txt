cmake_minimum_required(VERSION 3.10)
project(ParallelBlur)

set(CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED)

add_executable(blur main.cpp)
target_link_libraries(blur ${OpenCV_LIBS})
