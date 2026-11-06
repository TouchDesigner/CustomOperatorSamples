cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target.
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerPreProject.cmake)
project(SprinkleSOP LANGUAGES CXX)

# Shared TouchDesigner custom-operator build logic (repository root /cmake).
include(${CMAKE_CURRENT_LIST_DIR}/../../cmake/TouchDesignerCustomOP.cmake)

td_add_custom_op(SprinkleSOP
	SOURCES
		Parameters.cpp
		RandomPointsBuffer.cpp
		SprinkleSOP.cpp
		VolSprinkleTree.cpp
		CPlusPlus_Common.h
		GL_Extensions.h
		Parameters.h
		RandomPointsBuffer.h
		SOP_CPlusPlusBase.h
		SprinkleSOP.h
		VolSprinkleTree.h
)
