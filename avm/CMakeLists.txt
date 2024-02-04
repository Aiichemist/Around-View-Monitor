cmake_minimum_required(VERSION 3.0)

project(splicer LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

add_executable(splicer splicer.cpp)