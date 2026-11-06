cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(GeneratorDAT LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(GeneratorDAT
	SOURCES
		GeneratorDAT.cpp
		Parameters.cpp
		CPlusPlus_Common.h
		DAT_CPlusPlusBase.h
		GeneratorDAT.h
		Parameters.h
)
