# Creates the Python bindings for C++ code using pybind11
# Tim Jim, 21/04/2021

cmake_minimum_required(VERSION 3.18)
include(CheckIPOSupported)

project(kmac)  # output lib name

find_package(PythonInterp REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# Load pybind11
find_package(pybind11)  # Use this if installed in environment
# add_subdirectory(pybind11)  # Use this if pybind11 src is in this dir

# Add the C++ wrapper
pybind11_add_module(kmac SHARED
  kmac_wrapper.cpp
  ehvi_sliceupdate.cc ehvi_sliceupdate.h
  ehvi_multi.cc ehvi_multi.h
  ehvi_hvol.cc ehvi_hvol.h
  )

target_compile_features(kmac PRIVATE cxx_std_17)

# Add compilation optimization, if available
check_ipo_supported(RESULT supported OUTPUT error)
if( supported )
    message(STATUS "IPO / LTO enabled")
    set_property(TARGET kmac PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(STATUS "IPO / LTO not supported: <${error}>")
endif()

