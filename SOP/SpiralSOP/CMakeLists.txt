cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(SpiralSOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(SpiralSOP
	SOURCES
		Parameters.cpp
		SpiralSOP.cpp
		CPlusPlus_Common.h
		GL_Extensions.h
		Parameters.h
		SOP_CPlusPlusBase.h
		SpiralSOP.h
)
