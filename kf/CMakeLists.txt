cmake_minimum_required(VERSION 2.8)
project(kf)
set(KF_VERSION "0.0.1")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

set(CMAKE_PREFIX_PATH /home/bl/workspace/libs)


################################################################################
# Include CMake dependencies
################################################################################

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# Include helper macros and commands, and allow the included file to override
# the CMake policies in this file
include(${CMAKE_SOURCE_DIR}/cmake/CMakeHelper.cmake NO_POLICY_SCOPE)


################################################################################
# Options
################################################################################

option(OPENMP_ENABLED "Whether to enable OpenMP" ON)
option(DEBUG "ON for debug or OFF for release" OFF)

if (DEBUG)
  add_definitions(-DDEBUG)
endif ()


################################################################################
# Find packages
################################################################################

if (OPENMP_ENABLED)
  find_package(OpenMP QUIET)
endif ()

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)


################################################################################
# Compiler specific configuration
################################################################################

if (OPENMP_ENABLED AND OPENMP_FOUND)
  message(STATUS "Enabling OpenMP support")
  add_definitions("-DOPENMP_ENABLED")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else ()
  message(STATUS "Disabling OpenMP support")
endif ()

################################################################################
# Add sources
################################################################################

add_subdirectory(src)

KF_ADD_EXECUTABLE(main main.cpp)