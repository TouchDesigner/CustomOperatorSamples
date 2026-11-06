cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(ContoursTOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(ContoursTOP
	OPENCV_COMPONENTS core imgproc
	SOURCES
		ContoursTOP.cpp
		Parameters.cpp
		ContoursTOP.h
		CPlusPlus_Common.h
		Parameters.h
		TOP_CPlusPlusBase.h
)
