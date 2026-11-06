cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(FilterSOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(FilterSOP
	SOURCES
		FilterSOP.cpp
		Parameters.cpp
		CPlusPlus_Common.h
		FilterSOP.h
		GL_Extensions.h
		Parameters.h
		SOP_CPlusPlusBase.h
)
