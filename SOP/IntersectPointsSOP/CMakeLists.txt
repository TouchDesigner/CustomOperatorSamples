cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(IntersectPointsSOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(IntersectPointsSOP
	SOURCES
		IntersectPointsSOP.cpp
		Parameters.cpp
		CPlusPlus_Common.h
		GL_Extensions.h
		IntersectPointsSOP.h
		Parameters.h
		SOP_CPlusPlusBase.h
)
