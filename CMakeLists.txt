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
find_package(GLM REQUIRED)

add_library(viewer SHARED DRViewer.cpp glad.c widgets.cpp)
target_link_libraries(viewer ${GLFW3_LIBRARY} dl)

find_package(OpenCV 3)

if(OpenCV_FOUND)
    configure_file(data_list_path.h.in data_list_path.h)
    include_directories(${CMAKE_BINARY_DIR})
    include_directories(${OpenCV_INCLUDE_DIRS})
    add_executable(viewer_demo demo.cpp)
    target_link_libraries(viewer_demo viewer opencv_core opencv_imgcodecs opencv_imgproc)
else()
    message(WARNING "OpenCV not found, fail to build the demo")
endif()

install(TARGETS viewer LIBRARY DESTINATION lib)
install(FILES DRViewer.h DESTINATION include)
