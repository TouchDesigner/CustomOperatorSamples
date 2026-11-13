# TouchDesignerCUDAToolkit.cmake
#
# Picks the CUDA Toolkit that matches the CUDA major version TouchDesigner uses,
# when several toolkits are installed side by side. Without this, CMake takes
# whatever is the default: the last toolkit installed (CUDA_PATH) for Ninja and
# Makefiles, or the newest CUDA integration installed into Visual Studio.
#
# Must be included BEFORE the first project() call: the Visual Studio generator
# fixes its CUDA toolkit there, through the "cuda=" field of the generator
# toolset. Samples with .cu files include it, as does the root CMakeLists.txt.
#
# A toolkit chosen explicitly is always respected:
#   -DCMAKE_CUDA_COMPILER=<path to nvcc>    (Ninja, Makefiles)
#   -T cuda=<version or toolkit folder>     (Visual Studio)

include_guard(GLOBAL)

# CUDA major version TouchDesigner is built with. Plugins must be built with a
# toolkit of the same major version: each major version links its own CUDA
# runtime (CUDA 12: cudart64_12.dll; CUDA 13 on Windows: the runtime in the
# NVIDIA display driver, nvcudart_hybrid64.dll), and a plugin linked against
# another version's runtime either fails to load or doesn't match what
# TouchDesigner uses. Not a cache option on purpose: update it here when
# TouchDesigner moves to a new CUDA major version.
set(TD_CUDA_MAJOR_VERSION 13)

# Whether to build the samples that use CUDA. Turning this off keeps CUDA out of
# the build entirely: when CUDA is enabled, CMake's Visual Studio generator adds
# the CUDA build customization to every project in the solution, not just the
# CUDA ones, and with some Visual Studio / CUDA combinations (e.g. Visual Studio
# 2026 with CUDA 13.5) that makes IntelliSense fail for all of them. The CUDA
# samples can still be built in their own build folder, configured from a
# sample's folder. Defined here, before project(), so the toolkit selection
# below can be skipped too.
option(TD_BUILD_CUDA_SAMPLES "Build the samples that use CUDA (Windows only)" ON)

# TouchDesigner only supports CUDA on Windows.
if(NOT CMAKE_HOST_WIN32)
	return()
endif()

# CUDA samples turned off: don't give Visual Studio a CUDA toolset either.
if(NOT TD_BUILD_CUDA_SAMPLES)
	return()
endif()

# Returns the newest installed toolkit folder of TD_CUDA_MAJOR_VERSION, or "".
function(_td_find_cuda_toolkit out_var)
	# Candidates as "<version>|<folder>"; the version comes from the folder name
	# (vX.Y) or, for CUDA_PATH_VX_Y, from the variable name.
	set(candidates "")

	# Default install location: <Program Files>/NVIDIA GPU Computing Toolkit/CUDA/vX.Y
	if(DEFINED ENV{ProgramFiles})
		file(TO_CMAKE_PATH "$ENV{ProgramFiles}/NVIDIA GPU Computing Toolkit/CUDA" root)
		file(GLOB dirs LIST_DIRECTORIES true "${root}/v${TD_CUDA_MAJOR_VERSION}.*")
		foreach(dir IN LISTS dirs)
			get_filename_component(name "${dir}" NAME)
			if(name MATCHES "^v([0-9]+\\.[0-9]+)$")
				list(APPEND candidates "${CMAKE_MATCH_1}|${dir}")
			endif()
		endforeach()
	endif()

	# The installer also sets CUDA_PATH_VX_Y for every toolkit, wherever it is
	# installed (and CUDA_PATH for the last one installed, which is covered by
	# these as well).
	execute_process(COMMAND "${CMAKE_COMMAND}" -E environment
		OUTPUT_VARIABLE env ERROR_QUIET)
	string(REGEX MATCHALL "(^|\n)CUDA_PATH_V${TD_CUDA_MAJOR_VERSION}_[0-9]+=[^\r\n]*" vars "${env}")
	foreach(var IN LISTS vars)
		string(STRIP "${var}" var)
		if(var MATCHES "^CUDA_PATH_V([0-9]+)_([0-9]+)=(.*)$")
			# Save the matches first: the string(REGEX) below overwrites them.
			set(version "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
			file(TO_CMAKE_PATH "${CMAKE_MATCH_3}" dir)
			string(REGEX REPLACE "/+$" "" dir "${dir}")
			list(APPEND candidates "${version}|${dir}")
		endif()
	endforeach()

	# Newest version whose folder actually holds nvcc.
	list(REMOVE_DUPLICATES candidates)
	list(SORT candidates COMPARE NATURAL ORDER DESCENDING)
	set(${out_var} "" PARENT_SCOPE)
	foreach(candidate IN LISTS candidates)
		string(REGEX REPLACE "^[^|]*\\|" "" dir "${candidate}")
		if(EXISTS "${dir}/bin/nvcc.exe")
			set(${out_var} "${dir}" PARENT_SCOPE)
			break()
		endif()
	endforeach()
endfunction()

if(CMAKE_GENERATOR MATCHES "^Visual Studio")
	if(NOT CMAKE_GENERATOR_TOOLSET MATCHES "(^|,)cuda=")
		# Reuse the folder picked on the first configure, so installing another
		# toolkit later doesn't switch toolsets inside an existing build folder.
		if(NOT _TD_CUDA_TOOLKIT_DIR OR NOT EXISTS "${_TD_CUDA_TOOLKIT_DIR}/bin/nvcc.exe")
			_td_find_cuda_toolkit(_td_dir)
			set(_TD_CUDA_TOOLKIT_DIR "${_td_dir}" CACHE INTERNAL "")
		endif()
		if(_TD_CUDA_TOOLKIT_DIR)
			if(CMAKE_GENERATOR_TOOLSET)
				set(CMAKE_GENERATOR_TOOLSET "${CMAKE_GENERATOR_TOOLSET},cuda=${_TD_CUDA_TOOLKIT_DIR}")
			else()
				set(CMAKE_GENERATOR_TOOLSET "cuda=${_TD_CUDA_TOOLKIT_DIR}")
			endif()
			message(STATUS "Using CUDA toolkit ${_TD_CUDA_TOOLKIT_DIR} (override with -T cuda=<version or folder>)")
		endif()
	endif()
elseif(NOT CMAKE_CUDA_COMPILER)
	_td_find_cuda_toolkit(_td_dir)
	if(_td_dir)
		# Cached, so later configures keep using it.
		set(CMAKE_CUDA_COMPILER "${_td_dir}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler")
		message(STATUS "Using CUDA toolkit ${_td_dir} (override with -DCMAKE_CUDA_COMPILER=<path to nvcc>)")
	endif()
endif()
