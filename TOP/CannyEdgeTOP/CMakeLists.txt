cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target and CUDA Toolkit selection.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(CannyEdgeTOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_enable_cuda()

td_add_custom_op(CannyEdgeTOP
	WINDOWS_ONLY
	CUDA
	OPENCV_COMPONENTS core cudaarithm cudaimgproc
	SOURCES
		CannyEdgeTOP.cpp
		Parameters.cpp
		GpuUtils.cu
		CannyEdgeTOP.h
		CPlusPlus_Common.h
		Parameters.h
		TOP_CPlusPlusBase.h
		GpuUtils.cuh
)
