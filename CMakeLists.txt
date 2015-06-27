project(gifmm)
cmake_minimum_required(VERSION 2.8.7)

find_package(OpenCV REQUIRED)

include_directories(include)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++11 -g")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")

add_library(gdfmm SHARED
  src/expcache.cc
  src/gdfmm.cc)

add_executable(test
  src/test.cc)

target_link_libraries(gdfmm
  ${OpenCV_LIBRARIES})

target_link_libraries(test
  gdfmm)

