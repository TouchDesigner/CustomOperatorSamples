cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(BasicFilterCHOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(BasicFilterCHOP
	SOURCES
		BasicFilterCHOP.cpp
		Parameters.cpp
		BasicFilterCHOP.h
		CHOP_CPlusPlusBase.h
		CPlusPlus_Common.h
		Parameters.h
)
