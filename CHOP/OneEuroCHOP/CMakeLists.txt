cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(OneEuroCHOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(OneEuroCHOP
	SOURCES
		OneEuroCHOP.cpp
		OneEuroImpl.cpp
		Parameters.cpp
		CHOP_CPlusPlusBase.h
		CPlusPlus_Common.h
		OneEuroCHOP.h
		OneEuroImpl.h
		Parameters.h
)
