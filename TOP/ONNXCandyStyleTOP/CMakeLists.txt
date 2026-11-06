cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target and CUDA Toolkit selection.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(ONNXCandyStyleTOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_enable_cuda()

td_add_custom_op(ONNXCandyStyleTOP
	WINDOWS_ONLY
	CUDA
	ONNXRUNTIME
	SOURCES
		ONNXCandyStyleTOP.cpp
		kernel.cu
		CPlusPlus_Common.h
		ONNXCandyStyleTOP.h
		TOP_CPlusPlusBase.h
)
