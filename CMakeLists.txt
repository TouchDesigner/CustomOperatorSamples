cmake_minimum_required(VERSION 4.4)

# Before project(): macOS deployment target and CUDA Toolkit selection.
include(${CMAKE_CURRENT_LIST_DIR}/cmake/TouchDesignerPreProject.cmake)
project(CustomOperatorSamples LANGUAGES CXX)

# Builds every TouchDesigner Custom Operator Sample as one project/solution.
#
# Each sample folder also has its own CMakeLists.txt and can be configured on
# its own instead, e.g.:  cmake -S CHOP/BasicFilterCHOP -B CHOP/BasicFilterCHOP/build
#
# Samples whose dependencies (OpenCV, CUDA, ONNX Runtime) are not found, or that
# don't support the current platform, are skipped with a warning.

option(TD_BUILD_CHOPS "Build the CHOP samples" ON)
option(TD_BUILD_DATS  "Build the DAT samples"  ON)
option(TD_BUILD_SOPS  "Build the SOP samples"  ON)
option(TD_BUILD_TOPS  "Build the TOP samples"  ON)

# Tells td_add_custom_op() to skip (not fail on) samples with missing dependencies.
set(TD_CUSTOM_OPS_ROOT_BUILD TRUE)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/TouchDesignerCustomOP.cmake)

# CUDA must be enabled in the top-level directory for any sample to use it.
if(TD_BUILD_TOPS)
	td_enable_cuda()
endif()

if(TD_BUILD_CHOPS)
	add_subdirectory(CHOP/BasicFilterCHOP)
	add_subdirectory(CHOP/BasicGeneratorCHOP)
	add_subdirectory(CHOP/OneEuroCHOP)
	add_subdirectory(CHOP/TimeSliceFilterCHOP)
	add_subdirectory(CHOP/TimeSliceGeneratorCHOP)
endif()

if(TD_BUILD_DATS)
	add_subdirectory(DAT/FilterDAT)
	add_subdirectory(DAT/GeneratorDAT)
endif()

if(TD_BUILD_SOPS)
	add_subdirectory(SOP/FilterSOP)
	add_subdirectory(SOP/GeneratorSOP)
	add_subdirectory(SOP/IntersectPointsSOP)
	add_subdirectory(SOP/SpiralSOP)
	add_subdirectory(SOP/SprinkleSOP)
	add_subdirectory(SOP/WrapPointsSOP)
endif()

if(TD_BUILD_TOPS)
	add_subdirectory(TOP/BasicFilterTOP)
	add_subdirectory(TOP/CannyEdgeTOP)
	add_subdirectory(TOP/ContoursTOP)
	add_subdirectory(TOP/DistanceTransformTOP)
	add_subdirectory(TOP/ObjectDetectorTOP)
	add_subdirectory(TOP/ONNXCandyStyleTOP)
	add_subdirectory(TOP/OpticalFlowCPUTOP)
	add_subdirectory(TOP/SpectrumTOP)
endif()

get_property(_td_skipped GLOBAL PROPERTY TD_SKIPPED_SAMPLES)
if(_td_skipped)
	list(JOIN _td_skipped ", " _td_skipped)
	message(STATUS "Skipped samples (see warnings above): ${_td_skipped}")
endif()
