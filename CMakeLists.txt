cmake_minimum_required(VERSION 2.8)

project(viewer)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse -msse2 -msse3 -msse4 -msse4a -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g -ggdb")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/modules/")

find_package(Eigen3 REQUIRED)
find_package(GLFW3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} main.cpp DRViewer.cpp glad.c)
target_link_libraries(${CMAKE_PROJECT_NAME} ${GLFW3_LIBRARY} dl)
