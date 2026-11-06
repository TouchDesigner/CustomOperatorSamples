cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(FilterDAT LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(FilterDAT
	SOURCES
		FilterDAT.cpp
		Parameters.cpp
		CPlusPlus_Common.h
		DAT_CPlusPlusBase.h
		FilterDAT.h
		Parameters.h
)
