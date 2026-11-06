# TouchDesignerPreProject.cmake
#
# Settings that have to be in place BEFORE the first project() call, because
# project() is where CMake sets up the compilers and generator toolsets.
# Every sample's CMakeLists.txt (and the root one) includes this first:
#
#   cmake_minimum_required(VERSION 4.4)
#   include(${CMAKE_CURRENT_LIST_DIR}/<path to repository root>/cmake/TouchDesignerPreProject.cmake)
#   project(MySample LANGUAGES CXX)

include_guard(GLOBAL)

# Oldest macOS version the plugins run on; matches what TouchDesigner supports.
# A cache value, so -DCMAKE_OSX_DEPLOYMENT_TARGET=... still overrides it.
if(CMAKE_HOST_APPLE)
	set(CMAKE_OSX_DEPLOYMENT_TARGET "12.0" CACHE STRING "Minimum macOS version the plugins run on")
endif()

# Picks the CUDA Toolkit TouchDesigner needs when several are installed (the
# Visual Studio generator selects it in project()). Does nothing on macOS.
include(${CMAKE_CURRENT_LIST_DIR}/TouchDesignerCUDAToolkit.cmake)
