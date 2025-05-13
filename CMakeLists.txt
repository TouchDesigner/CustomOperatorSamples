cmake_minimum_required(VERSION 3.10)
project(CustomOperators)

set(OP_TYPES_VALID "CHOP" "SOP" "DAT" "TOP")
set(OP_TYPES "CHOP" CACHE STRING "Operator families to build (CHOP,TOP,SOP,DAT)")
set_property(CACHE OP_TYPES PROPERTY STRINGS ${OP_TYPES_VALID})

# Validate input
if(NOT OP_TYPES IN_LIST OP_TYPES_VALID)
    message(FATAL_ERROR "Invalid OP_TYPES='${OP_TYPES}'. Must be one of: ${OP_TYPES_VALID}")
endif()

message(STATUS "Building operator families: ${OP_TYPES}")

# CHOPs
if("CHOP" STREQUAL OP_TYPES)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/CHOP/BasicFilterCHOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/CHOP/BasicGeneratorCHOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/CHOP/OneEuroCHOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/CHOP/TimeSliceFilterCHOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/CHOP/TimeSliceGeneratorCHOP)
	message(STATUS "-------------------------")
endif()

# DATs
if("DAT" STREQUAL OP_TYPES)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DAT/FilterDAT)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/DAT/GeneratorDAT)
	message(STATUS "-------------------------")
endif()

# SOPs
if("SOP" STREQUAL OP_TYPES)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/SOP/FilterSOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/SOP/GeneratorSOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/SOP/IntersectPointsSOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/SOP/SpiralSOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/SOP/SprinkleSOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/SOP/WrapPointsSOP)
	message(STATUS "-------------------------")
endif()

# TOPs
if("TOP" STREQUAL OP_TYPES)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TOP/BasicFilterTOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TOP/DistanceTransformTOP)
	add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/TOP/ObjectDetectorTOP)
	message(STATUS "-------------------------")
endif()