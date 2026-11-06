cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(OpticalFlowCPUTOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(OpticalFlowCPUTOP
	OPENCV_COMPONENTS core imgproc video tracking
	SOURCES
		OpticalFlowCPUTOP.cpp
		Parameters.cpp
		CPlusPlus_Common.h
		OpticalFlowCPUTOP.h
		Parameters.h
		TOP_CPlusPlusBase.h
)
