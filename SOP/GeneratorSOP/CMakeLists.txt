cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(GeneratorSOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(GeneratorSOP
	SOURCES
		GeneratorSOP.cpp
		Parameters.cpp
		ShapeGenerator.cpp
		CPlusPlus_Common.h
		GeneratorSOP.h
		GL_Extensions.h
		Parameters.h
		ShapeGenerator.h
		SOP_CPlusPlusBase.h
)
