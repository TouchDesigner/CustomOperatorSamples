cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(BasicFilterTOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(BasicFilterTOP
	SOURCES
		BasicFilterTOP.cpp
		FilterWork.cpp
		Parameters.cpp
		ThreadManager.cpp
		BasicFilterTOP.h
		CPlusPlus_Common.h
		FilterWork.h
		Parameters.h
		ThreadManager.h
		TOP_CPlusPlusBase.h
)
