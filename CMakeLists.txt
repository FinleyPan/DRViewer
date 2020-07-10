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

find_package(GLFW3 REQUIRED)

add_library(viewer SHARED DRViewer.cpp glad.c widgets.cpp)
target_link_libraries(viewer ${GLFW3_LIBRARY} dl)

add_executable(viewer-test main.cpp)
target_link_libraries(viewer-test viewer)

install(TARGETS viewer LIBRARY DESTINATION lib)
install(FILES DRViewer.h DESTINATION include)
install(FILES glad.h DESTINATION include)
