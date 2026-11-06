cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target and CUDA Toolkit selection.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(SpectrumTOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_enable_cuda()

td_add_custom_op(SpectrumTOP
	WINDOWS_ONLY
	CUDA
	OPENCV_COMPONENTS core imgproc video tracking cudaarithm cudaimgproc cudawarping
	SOURCES
		Parameters.cpp
		SpectrumTOP.cpp
		GpuUtils.cu
		CPlusPlus_Common.h
		Parameters.h
		SpectrumTOP.h
		TOP_CPlusPlusBase.h
		GpuUtils.cuh
)
