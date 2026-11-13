# TouchDesignerCustomOP.cmake
#
# Shared build logic for the TouchDesigner Custom Operator Samples.
#
# Every sample's CMakeLists.txt includes TouchDesignerPreProject.cmake before
# project(), then this file, and calls td_add_custom_op(). The same
# CMakeLists.txt works both when configured on its own
# (e.g. `cmake -S CHOP/BasicFilterCHOP -B build`) and when pulled in by the
# repository's root CMakeLists.txt, which builds every sample in one project.
#
# Usage:
#
#   td_enable_cuda()   # only for samples with .cu files; call at file scope
#
#   td_add_custom_op(<target>
#       SOURCES <files...>                 # .cpp/.h/.cu/.cuh, relative to the sample folder
#       [INCLUDE_DIRECTORIES <dirs...>]    # e.g. include, if the TD headers live there
#       [CXX_STANDARD <17|20|...>]         # default 17
#       [FAMILY <CHOP|TOP|...>]            # IDE folder; default: name of the parent folder
#       [OPENCV_COMPONENTS <modules...>]   # e.g. core imgproc cudaarithm
#       [CUDA]                             # compile .cu files and link the CUDA runtime
#       [PYTHON]                           # build against TouchDesigner's Python
#       [ONNXRUNTIME]                      # link the ONNX Runtime C API
#       [WINDOWS_ONLY]                     # sample only builds on Windows
#       [LINK_LIBRARIES <libs...>])
#
#   td_skip_custom_op(<target> <reason>)
#       For a dependency only one sample needs: skips the sample with a warning
#       when building all samples, or stops with an error when building it on
#       its own. Call it instead of td_add_custom_op() and return().
#
# On macOS, a sample whose CPlusPlus_Common.h imports <Metal/Metal.h> (newer
# TouchDesigner headers) has its .cpp files compiled as Objective-C++ and is
# linked with the Foundation and Metal frameworks automatically.
#
# Cache variables you may want to set (all optional):
#
#   TD_PLUGIN_OUTPUT_DIR      Where the built plugin (.dll / .plugin) is written.
#                             Defaults to <build folder>/plugins.
#   CMAKE_INSTALL_PREFIX      Where `cmake --install` copies the plugins. Defaults to
#                             TouchDesigner's system-wide Plugins folder:
#                               Windows: <Documents>/Derivative/Plugins
#                               macOS:   ~/Library/Application Support/Derivative/TouchDesigner099/Plugins
#   TD_INSTALLATION_PATH      Root of the TouchDesigner installation:
#                               Windows: the install folder,
#                                        default C:/Program Files/Derivative/TouchDesigner
#                               macOS:   the application bundle,
#                                        default /Applications/TouchDesigner.app
#                             The bundled 3rdParty headers/libraries, Python,
#                             the macOS Frameworks folder and the debugger
#                             executable are all found relative to it.
#   TD_USE_BUNDLED_OPENCV     Link the OpenCV that ships with TouchDesigner
#                             (default ON). Turn OFF to use find_package(OpenCV).
#   ONNXRUNTIME_ROOT          Root of an extracted ONNX Runtime package.
#   TD_EXECUTABLE             TouchDesigner executable used as the debugger target.
#                             Empty (default) = derived from TD_INSTALLATION_PATH.
#   TD_CUDA_ALLOW_UNSUPPORTED_COMPILER
#                             Pass -allow-unsupported-compiler to nvcc (needed for
#                             CUDA < 12.4 with newer Visual Studio 2022 toolsets).

include_guard(GLOBAL)

# ----------------------------------------------------------------------------
# Options / cache variables
# ----------------------------------------------------------------------------

# Filled in with the actual default so it is visible in cmake-gui / ccmake and
# the CMake cache. Clearing it restores the default on the next configure.
set(_td_plugin_output_default "${CMAKE_BINARY_DIR}/plugins")
set(TD_PLUGIN_OUTPUT_DIR "${_td_plugin_output_default}" CACHE PATH
	"Folder the built plugins are written to (default: <build folder>/plugins)")
if(NOT TD_PLUGIN_OUTPUT_DIR)
	set_property(CACHE TD_PLUGIN_OUTPUT_DIR PROPERTY VALUE "${_td_plugin_output_default}")
endif()

# `cmake --install` copies the plugins into TouchDesigner's system-wide Plugins
# folder, which TouchDesigner scans (including subfolders) when it starts.
# Only replace CMake's default prefix, so -DCMAKE_INSTALL_PREFIX=... and
# `cmake --install <build> --prefix <dir>` still work.
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
	set(_td_install_default "")
	if(WIN32)
		# Ask the shell for the real Documents folder, which may be redirected
		# (e.g. to OneDrive) rather than living at %USERPROFILE%\Documents.
		execute_process(
			COMMAND powershell -NoProfile -NonInteractive -Command
				"[Environment]::GetFolderPath('MyDocuments')"
			OUTPUT_VARIABLE _td_documents
			OUTPUT_STRIP_TRAILING_WHITESPACE
			ERROR_QUIET)
		if(NOT _td_documents AND DEFINED ENV{USERPROFILE})
			set(_td_documents "$ENV{USERPROFILE}/Documents")
		endif()
		if(_td_documents)
			file(TO_CMAKE_PATH "${_td_documents}/Derivative/Plugins" _td_install_default)
		endif()
	elseif(APPLE AND DEFINED ENV{HOME})
		set(_td_install_default "$ENV{HOME}/Library/Application Support/Derivative/TouchDesigner099/Plugins")
	endif()
	if(_td_install_default)
		set(CMAKE_INSTALL_PREFIX "${_td_install_default}" CACHE PATH
			"Install folder for the plugins (TouchDesigner's Plugins folder)" FORCE)
	endif()
endif()

# ---- TouchDesigner installation ---------------------------------------------
# OpenCV major version the samples are written for, and that the TouchDesigner
# build they target ships (headers in Samples/CPlusPlus/3rdParty, libraries
# loaded at runtime). Plugins must be compiled against the same OpenCV that
# TouchDesigner loads. Not a cache option on purpose: update it here when the
# samples move to a TouchDesigner build with a new OpenCV major version.
set(TD_OPENCV_MAJOR_VERSION 4)

# Python version the TouchDesigner build ships (major.minor), and whether it is
# the free-threaded build (3.14t etc.). Samples using Python (td_add_custom_op's
# PYTHON option) are built against the Python in the installation, and
# configuring stops if that isn't this version, which catches TD_INSTALLATION_PATH
# pointing at another series. Not a cache option on purpose: update it here
# when the samples move to a TouchDesigner build with a new Python.
set(TD_PYTHON_VERSION "3.11")
set(TD_PYTHON_FREE_THREADED FALSE)

# TouchDesigner build series these samples target (the series that ships the
# CUDA, OpenCV and Python versions above), as it reads in messages:
# "the latest TouchDesigner <series> series build". Only used in messages that
# tell people which installation to point TD_INSTALLATION_PATH at.
set(TD_BUILD_SERIES "2025")
if(APPLE)
	set(_td_installation_hint "Set TD_INSTALLATION_PATH to the TouchDesigner.app of the latest TouchDesigner ${TD_BUILD_SERIES} series build.")
else()
	set(_td_installation_hint "Set TD_INSTALLATION_PATH to the installation folder of the latest TouchDesigner ${TD_BUILD_SERIES} series build.")
endif()

# TD_INSTALLATION_PATH is the one setting for where TouchDesigner is installed;
# everything taken from the install is derived from it below.
if(WIN32)
	set(_td_installation_default "C:/Program Files/Derivative/TouchDesigner")
elseif(APPLE)
	set(_td_installation_default "/Applications/TouchDesigner.app")
else()
	set(_td_installation_default "")
endif()

set(TD_INSTALLATION_PATH "${_td_installation_default}" CACHE PATH
	"Root of the TouchDesigner installation (install folder on Windows, TouchDesigner.app on macOS)")
file(TO_CMAKE_PATH "${TD_INSTALLATION_PATH}" TD_INSTALLATION_PATH)

# Derived from TD_INSTALLATION_PATH on every configure (not cached), so changing the
# root is enough to point everything at another installation.
set(TD_3RDPARTY_PATH "")
set(TD_FRAMEWORKS_PATH "")
set(_td_default_executable "")
if(TD_INSTALLATION_PATH)
	if(WIN32)
		set(TD_3RDPARTY_PATH "${TD_INSTALLATION_PATH}/Samples/CPlusPlus/3rdParty")
		set(_td_default_executable "${TD_INSTALLATION_PATH}/bin/TouchDesigner.exe")
	elseif(APPLE)
		set(TD_3RDPARTY_PATH "${TD_INSTALLATION_PATH}/Contents/Resources/tfs/Samples/CPlusPlus/3rdParty")
		set(TD_FRAMEWORKS_PATH "${TD_INSTALLATION_PATH}/Contents/Frameworks")
		set(_td_default_executable "${TD_INSTALLATION_PATH}")
	endif()
endif()

set(TD_EXECUTABLE "" CACHE FILEPATH
	"TouchDesigner executable launched when debugging a plugin. Empty = derived from TD_INSTALLATION_PATH")
if(TD_EXECUTABLE)
	set(_td_debug_executable "${TD_EXECUTABLE}")
else()
	set(_td_debug_executable "${_td_default_executable}")
endif()

get_property(_td_install_reported GLOBAL PROPERTY _TD_INSTALL_REPORTED)
if(NOT _td_install_reported AND (WIN32 OR APPLE))
	set_property(GLOBAL PROPERTY _TD_INSTALL_REPORTED TRUE)
	message(STATUS "TouchDesigner installation: ${TD_INSTALLATION_PATH}")
	if(NOT IS_DIRECTORY "${TD_INSTALLATION_PATH}")
		message(WARNING "TD_INSTALLATION_PATH '${TD_INSTALLATION_PATH}' does not exist. ${_td_installation_hint} Samples that use its bundled libraries or Python need it.")
	endif()
endif()

# ---- Other settings -------------------------------------------------------
if(WIN32 OR APPLE)
	option(TD_USE_BUNDLED_OPENCV "Link against the OpenCV bundled with TouchDesigner" ON)
endif()

if(WIN32)
	set(ONNXRUNTIME_ROOT "C:/Program Files/onnxruntime" CACHE PATH
		"Root folder of the ONNX Runtime package (contains include/ and lib/)")
	option(TD_CUDA_ALLOW_UNSUPPORTED_COMPILER
		"Pass -allow-unsupported-compiler to nvcc (CUDA < 12.4 with newer MSVC)" OFF)
else()
	set(ONNXRUNTIME_ROOT "" CACHE PATH
		"Root folder of the ONNX Runtime package (contains include/ and lib/)")
endif()

if(APPLE)
	set(TD_BUNDLE_IDENTIFIER_PREFIX "ca.derivative.samples" CACHE STRING
		"Prefix for the CFBundleIdentifier of the generated .plugin bundles")
endif()

# Group targets into folders (CHOP/SOP/TOP/DAT) in Visual Studio and Xcode.
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# macOS: newer TouchDesigner headers (CPlusPlus_Common.h) import <Metal/Metal.h>,
# so samples using them are compiled as Objective-C++ (see td_add_custom_op).
# Like CUDA, the language has to be enabled at directory scope; this file is
# included right after project().
if(APPLE)
	enable_language(OBJCXX)
endif()

# Xcode: a scheme for every target, including ALL_BUILD, install and
# ZERO_CHECK, and each plugin (so Product > Archive can be run on one plugin).
# This has to be all or nothing: as soon as CMake generates any scheme, it
# turns off Xcode's automatic scheme creation for the project, which is the
# only other way ALL_BUILD and install would get one. Set here (the top-level
# scope in the all-samples build) so the built-in targets see it too.
set(CMAKE_XCODE_GENERATE_SCHEME ON)

# ----------------------------------------------------------------------------
# td_enable_cuda()
#
# Enables the CUDA language if a CUDA toolkit is available. This is a macro
# (not a function) because enable_language() must run at file scope.
# Sets TD_CUDA_AVAILABLE to TRUE/FALSE.
# ----------------------------------------------------------------------------
# TD_CUDA_MAJOR_VERSION, and picking the matching toolkit when several are
# installed. Samples with .cu files include this before project() already (the
# Visual Studio generator needs that); for everything else this is enough.
include(${CMAKE_CURRENT_LIST_DIR}/TouchDesignerCUDAToolkit.cmake)

# Internal: the "major.minor" version of an nvcc, without enabling CUDA.
# Cached per compiler path so it only runs nvcc once per build folder.
function(_td_nvcc_version nvcc out_var)
	if(NOT _TD_NVCC_VERSION_FOR STREQUAL nvcc)
		execute_process(COMMAND "${nvcc}" --version
			OUTPUT_VARIABLE out ERROR_QUIET RESULT_VARIABLE rc)
		set(version "")
		if(rc EQUAL 0 AND out MATCHES "release ([0-9]+\\.[0-9]+)")
			set(version "${CMAKE_MATCH_1}")
		endif()
		set(_TD_NVCC_VERSION "${version}" CACHE INTERNAL "")
		set(_TD_NVCC_VERSION_FOR "${nvcc}" CACHE INTERNAL "")
	endif()
	set(${out_var} "${_TD_NVCC_VERSION}" PARENT_SCOPE)
endfunction()

macro(td_enable_cuda)
	set(TD_CUDA_ERROR "")
	if(NOT WIN32)
		# TouchDesigner only supports CUDA on Windows.
		set(TD_CUDA_AVAILABLE FALSE)
	elseif(NOT TD_BUILD_CUDA_SAMPLES)
		# Turned off (see TouchDesignerCUDAToolkit.cmake): CUDA is never enabled,
		# so no project in the solution gets the CUDA build customization.
		set(TD_CUDA_AVAILABLE FALSE)
		set(TD_CUDA_ERROR "the CUDA samples are turned off (TD_BUILD_CUDA_SAMPLES is OFF)")
	elseif(TD_CUDA_MAJOR_VERSION VERSION_LESS 13
			AND (CMAKE_GENERATOR MATCHES "^Visual Studio 18 " OR (MSVC AND MSVC_VERSION GREATER_EQUAL 1950)))
		# CUDA 12 doesn't support Visual Studio 2026: its nvcc doesn't accept
		# the VS 2026 compiler (MSVC 19.50+, toolset v145), and its Visual Studio
		# integration only installs into VS 2022. Checked before CUDA is enabled,
		# so this combination never gets as far as a failing compiler test.
		set(TD_CUDA_AVAILABLE FALSE)
		if(CMAKE_GENERATOR MATCHES "^Visual Studio 18 ")
			set(_td_cuda_vs_found "the ${CMAKE_GENERATOR} generator")
		else()
			set(_td_cuda_vs_found "the Visual Studio 2026 compiler (MSVC_VERSION ${MSVC_VERSION})")
		endif()
		set(TD_CUDA_ERROR "CUDA ${TD_CUDA_MAJOR_VERSION}, which TouchDesigner ${TD_BUILD_SERIES} series builds use, doesn't support Visual Studio 2026, but this build uses ${_td_cuda_vs_found}. Build the CUDA samples with Visual Studio 2022: configure a fresh build folder with -G \"Visual Studio 17 2022\", or use Ninja from a Visual Studio 2022 developer prompt.")
	else()
		# check_language() caches CMAKE_CUDA_COMPILER, so it only does work on
		# the first configure. enable_language() below must still run on every
		# configure: a language enabled in an earlier run is not remembered, and
		# without it .cu files are silently left out of the build.
		if(NOT CMAKE_CUDA_COMPILER)
			include(CheckLanguage)
			check_language(CUDA)
		endif()
		if(CMAKE_CUDA_COMPILER)
			# Check the toolkit's version before enabling CUDA: a toolkit of
			# another major version may not even accept the default
			# architectures below, which would fail with a confusing error.
			_td_nvcc_version("${CMAKE_CUDA_COMPILER}" _td_cuda_found_version)
			if(_td_cuda_found_version AND NOT _td_cuda_found_version MATCHES "^${TD_CUDA_MAJOR_VERSION}\\.")
				set(TD_CUDA_AVAILABLE FALSE)
			else()
				if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
					# Real code for Pascal..Hopper plus PTX for older (JIT) and future GPUs.
					set(CMAKE_CUDA_ARCHITECTURES "50-virtual;60;61;70;75;80;86;89;90;90-virtual")
				endif()
				# The macro runs once per directory and subdirectories inherit the
				# flags, so only append them once.
				if(TD_CUDA_ALLOW_UNSUPPORTED_COMPILER AND NOT CMAKE_CUDA_FLAGS MATCHES "-allow-unsupported-compiler")
					string(APPEND CMAKE_CUDA_FLAGS " -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
				endif()
				enable_language(CUDA)
				if(NOT TARGET CUDA::cudart)
					find_package(CUDAToolkit)
				endif()
				set(TD_CUDA_AVAILABLE TRUE)

				# The compiler version is what CMake actually builds with; with the
				# Visual Studio generator that is the VS CUDA integration, which can
				# differ from CUDA_PATH. Check the toolkit's libraries too.
				set(_td_cuda_found_version "${CMAKE_CUDA_COMPILER_VERSION}")
				if(NOT _td_cuda_found_version MATCHES "^${TD_CUDA_MAJOR_VERSION}\\.")
					set(TD_CUDA_AVAILABLE FALSE)
				elseif(DEFINED CUDAToolkit_VERSION_MAJOR AND NOT CUDAToolkit_VERSION_MAJOR STREQUAL TD_CUDA_MAJOR_VERSION)
					set(_td_cuda_found_version "${CUDAToolkit_VERSION}")
					set(TD_CUDA_AVAILABLE FALSE)
				endif()
			endif()
			if(NOT TD_CUDA_AVAILABLE)
				set(TD_CUDA_ERROR "found CUDA ${_td_cuda_found_version}, but TouchDesigner ${TD_BUILD_SERIES} series builds use CUDA ${TD_CUDA_MAJOR_VERSION}.x and only ship that runtime. Install a CUDA ${TD_CUDA_MAJOR_VERSION}.x toolkit and select it in a fresh build folder with -DCMAKE_CUDA_COMPILER=<path to its nvcc> (Visual Studio generator: -T cuda=${TD_CUDA_MAJOR_VERSION}.x)")
			endif()
		else()
			set(TD_CUDA_AVAILABLE FALSE)
		endif()
	endif()

	# Visual Studio 2026: CUDA's build customization is imported into every
	# project of the solution, and can make IntelliSense fail for all of them.
	get_property(_td_cuda_vs_note GLOBAL PROPERTY _TD_CUDA_VS2026_NOTE)
	if(TD_CUDA_AVAILABLE AND TD_CUSTOM_OPS_ROOT_BUILD AND NOT _td_cuda_vs_note
			AND CMAKE_GENERATOR MATCHES "^Visual Studio 18 ")
		set_property(GLOBAL PROPERTY _TD_CUDA_VS2026_NOTE TRUE)
		message(NOTICE "Note: CUDA is enabled for this ${CMAKE_GENERATOR} solution, so CMake adds the CUDA build customization to every project in it. If Visual Studio reports \"Designtime build failed\" (IntelliSense unavailable) for all projects, configure this build folder with -DTD_BUILD_CUDA_SAMPLES=OFF, and build the CUDA samples in a build folder of their own, configured from the sample's folder.")
	endif()
endmacro()

# ----------------------------------------------------------------------------
# Internal: locate OpenCV.
# Sets <prefix>_FOUND, <prefix>_INCLUDE_DIRS, <prefix>_LIBRARIES, <prefix>_VERSION,
# <prefix>_BUNDLED (TRUE when TouchDesigner's copy is used) and <prefix>_ERROR.
# ----------------------------------------------------------------------------

# Internal: read "major;minor;revision" from <include_dir>/opencv2/core/version.hpp.
function(_td_read_opencv_version include_dir out_var)
	set(version "")
	set(file "${include_dir}/opencv2/core/version.hpp")
	if(EXISTS "${file}")
		file(STRINGS "${file}" defines REGEX "^#define[ \t]+CV_VERSION_(MAJOR|MINOR|REVISION)[ \t]+[0-9]+")
		foreach(part MAJOR MINOR REVISION)
			set(value "")
			foreach(define IN LISTS defines)
				if(define MATCHES "^#define[ \t]+CV_VERSION_${part}[ \t]+([0-9]+)")
					set(value "${CMAKE_MATCH_1}")
				endif()
			endforeach()
			if(value STREQUAL "")
				set(version "")
				break()
			endif()
			list(APPEND version "${value}")
		endforeach()
	endif()
	set(${out_var} "${version}" PARENT_SCOPE)
endfunction()

function(_td_find_opencv prefix)
	set(components ${ARGN})
	set(found FALSE)
	set(bundled FALSE)
	set(includes "")
	set(libs "")
	set(version "")
	set(error "")
	# Set when TouchDesigner's OpenCV is present but unusable: falling back to
	# another OpenCV would then build against a different OpenCV than the one
	# TouchDesigner loads.
	set(no_fallback FALSE)

	# Prefer the OpenCV that ships with TouchDesigner, so the plugin is built
	# against the same headers and libraries TouchDesigner loads at runtime.
	if((WIN32 OR APPLE) AND TD_USE_BUNDLED_OPENCV)
		# Headers: <3rdParty>/opencv/include on Windows, .../include/opencv<N> on
		# macOS. Look at every opencv<N> folder, so an install with another major
		# version is reported as such rather than as missing.
		set(cv_inc "")
		file(GLOB cv_inc_versioned LIST_DIRECTORIES true "${TD_3RDPARTY_PATH}/opencv/include/opencv[0-9]*")
		list(SORT cv_inc_versioned COMPARE NATURAL ORDER DESCENDING)
		foreach(dir "${TD_3RDPARTY_PATH}/opencv/include/opencv${TD_OPENCV_MAJOR_VERSION}"
				${cv_inc_versioned}
				"${TD_3RDPARTY_PATH}/opencv/include")
			if(EXISTS "${dir}/opencv2/core/version.hpp")
				set(cv_inc "${dir}")
				break()
			endif()
		endforeach()
		if(WIN32)
			set(cv_lib "${TD_3RDPARTY_PATH}/opencv/lib/Win64")
		else()
			set(cv_lib "${TD_FRAMEWORKS_PATH}")
		endif()

		if(cv_inc AND IS_DIRECTORY "${cv_lib}")
			_td_read_opencv_version("${cv_inc}" cv_version)
			list(JOIN cv_version "." cv_version_string)
			if(NOT cv_version)
				set(no_fallback TRUE)
				set(error "Could not read the OpenCV version from '${cv_inc}/opencv2/core/version.hpp' (looking for OpenCV ${TD_OPENCV_MAJOR_VERSION}.x). ${_td_installation_hint} ")
			else()
				list(GET cv_version 0 cv_major)
				list(GET cv_version 1 cv_minor)
				list(GET cv_version 2 cv_revision)
				if(NOT cv_major STREQUAL TD_OPENCV_MAJOR_VERSION)
					set(no_fallback TRUE)
					set(error "TouchDesigner in '${TD_INSTALLATION_PATH}' ships OpenCV ${cv_version_string}, but these samples are looking for OpenCV ${TD_OPENCV_MAJOR_VERSION}.x, which ships with TouchDesigner ${TD_BUILD_SERIES} series builds. ${_td_installation_hint} ")
				else()
					set(found TRUE)
					set(bundled TRUE)
					set(includes "${cv_inc}")
					set(version "${cv_version_string}")
					# The library names carry the version, so only libraries
					# that match the headers are used:
					#   Windows: opencv_core4110.lib
					#   macOS:   libopencv_core.4.11.0.dylib, libopencv_core.411.dylib
					string(REGEX MATCH "[0-9][0-9]$" cv_minor_2digits "00${cv_minor}")
					foreach(comp IN LISTS components)
						if(WIN32)
							set(names "opencv_${comp}${cv_major}${cv_minor}${cv_revision}.lib"
								"opencv_world${cv_major}${cv_minor}${cv_revision}.lib")
						else()
							set(names "libopencv_${comp}.${cv_major}.${cv_minor}.${cv_revision}.dylib"
								"libopencv_${comp}.${cv_major}${cv_minor_2digits}.dylib")
						endif()
						set(lib "")
						foreach(name IN LISTS names)
							if(EXISTS "${cv_lib}/${name}")
								set(lib "${cv_lib}/${name}")
								break()
							endif()
						endforeach()
						if(lib)
							list(APPEND libs "${lib}")
						else()
							set(found FALSE)
							set(no_fallback TRUE)
							list(GET names 0 expected)
							string(APPEND error "${expected} (OpenCV ${cv_version_string}, matching the headers) not found in ${cv_lib}; the TouchDesigner installation looks incomplete. ")
						endif()
					endforeach()
					list(REMOVE_DUPLICATES libs)
				endif()
			endif()
		elseif(WIN32)
			set(error "TouchDesigner's OpenCV (looking for OpenCV ${TD_OPENCV_MAJOR_VERSION}.x) was not found in '${TD_3RDPARTY_PATH}'. ${_td_installation_hint} (Or set TD_USE_BUNDLED_OPENCV=OFF and OpenCV_DIR to an OpenCV ${TD_OPENCV_MAJOR_VERSION}.x build.) ")
		else()
			set(error "TouchDesigner's OpenCV (looking for OpenCV ${TD_OPENCV_MAJOR_VERSION}.x) was not found in '${TD_INSTALLATION_PATH}'. ${_td_installation_hint} (Or set TD_USE_BUNDLED_OPENCV=OFF and OpenCV_DIR to an OpenCV ${TD_OPENCV_MAJOR_VERSION}.x build.) ")
		endif()
	endif()

	# Otherwise (and as a fallback when TouchDesigner's OpenCV isn't there at
	# all): a regular OpenCV install, e.g. Homebrew's `brew install opencv`, or
	# OpenCV_DIR pointing at an OpenCV build. It must be the same major version.
	if(NOT found AND NOT no_fallback)
		find_package(OpenCV QUIET COMPONENTS ${components})
		if(OpenCV_FOUND AND NOT OpenCV_VERSION_MAJOR STREQUAL TD_OPENCV_MAJOR_VERSION)
			string(APPEND error "find_package(OpenCV) found OpenCV ${OpenCV_VERSION} in '${OpenCV_DIR}', but these samples are looking for OpenCV ${TD_OPENCV_MAJOR_VERSION}.x. Set OpenCV_DIR to an OpenCV ${TD_OPENCV_MAJOR_VERSION}.x build.")
		elseif(OpenCV_FOUND)
			set(found TRUE)
			set(bundled FALSE)
			set(error "")
			set(includes ${OpenCV_INCLUDE_DIRS})
			set(libs ${OpenCV_LIBS})
			set(version "${OpenCV_VERSION}")
		else()
			string(APPEND error "find_package(OpenCV COMPONENTS ${components}) found no OpenCV (looking for OpenCV ${TD_OPENCV_MAJOR_VERSION}.x; set OpenCV_DIR to an OpenCV ${TD_OPENCV_MAJOR_VERSION}.x build).")
		endif()
	endif()

	set(${prefix}_FOUND ${found} PARENT_SCOPE)
	set(${prefix}_BUNDLED ${bundled} PARENT_SCOPE)
	set(${prefix}_INCLUDE_DIRS "${includes}" PARENT_SCOPE)
	set(${prefix}_LIBRARIES "${libs}" PARENT_SCOPE)
	set(${prefix}_VERSION "${version}" PARENT_SCOPE)
	set(${prefix}_ERROR "${error}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Internal: locate TouchDesigner's Python, to build against the interpreter the
# plugin is loaded into.
#   Windows, 2026 series builds: headers in <TD_INSTALLATION_PATH>/bin/include,
#            import library in <TD_INSTALLATION_PATH>/bin/libs
#   Windows, 2025 series builds: headers in Samples/CPlusPlus/3rdParty/Python/Include
#            and Include/PC (pyconfig.h, as in a CPython source tree),
#            import library in Samples/CPlusPlus/3rdParty/Python/lib/x64
#   macOS:   <TD_INSTALLATION_PATH>/Contents/Frameworks/Python.framework
# Sets <prefix>_FOUND, <prefix>_INCLUDE_DIRS, <prefix>_LIBRARIES,
# <prefix>_LIBRARY_DIRS, <prefix>_DEFINITIONS, <prefix>_VERSION (e.g. 3.14),
# <prefix>_FREE_THREADED and <prefix>_ERROR.
# ----------------------------------------------------------------------------
function(_td_find_python prefix)
	set(found FALSE)
	set(includes "")
	set(libs "")
	set(lib_dirs "")
	set(defs "")
	set(version "")
	set(free_threaded FALSE)
	set(error "")

	# Header folder candidates, each paired with its import library folder
	# (Windows only).
	set(inc_candidates "")
	set(lib_dirs_for "")
	if(WIN32)
		# 2026 series layout.
		list(APPEND inc_candidates "${TD_INSTALLATION_PATH}/bin/include")
		list(APPEND lib_dirs_for "${TD_INSTALLATION_PATH}/bin/libs")
		# 2025 series layout.
		list(APPEND inc_candidates "${TD_3RDPARTY_PATH}/Python/Include")
		list(APPEND lib_dirs_for "${TD_3RDPARTY_PATH}/Python/lib/x64")
	elseif(APPLE)
		set(framework "${TD_FRAMEWORKS_PATH}/Python.framework")
		# Headers is a symlink to Versions/Current/Headers in a standard framework;
		# also look in the version folders directly in case it is missing.
		file(GLOB versioned_incs LIST_DIRECTORIES true "${framework}/Versions/*/include/python3*")
		list(SORT versioned_incs COMPARE NATURAL ORDER DESCENDING)
		set(inc_candidates "${framework}/Headers" ${versioned_incs})
	else()
		set(error "TouchDesigner's Python is only available on Windows and macOS.")
	endif()

	set(inc "")
	set(lib_dir "")
	set(index 0)
	foreach(dir IN LISTS inc_candidates)
		if(EXISTS "${dir}/Python.h" AND EXISTS "${dir}/patchlevel.h")
			set(inc "${dir}")
			if(lib_dirs_for)
				list(GET lib_dirs_for ${index} lib_dir)
			endif()
			break()
		endif()
		math(EXPR index "${index} + 1")
	endforeach()

	if(inc)
		file(STRINGS "${inc}/patchlevel.h" lines REGEX "^#define[ \t]+PY_(MAJOR|MINOR)_VERSION[ \t]+[0-9]+")
		set(major "")
		set(minor "")
		foreach(line IN LISTS lines)
			if(line MATCHES "PY_MAJOR_VERSION[ \t]+([0-9]+)")
				set(major "${CMAKE_MATCH_1}")
			elseif(line MATCHES "PY_MINOR_VERSION[ \t]+([0-9]+)")
				set(minor "${CMAKE_MATCH_1}")
			endif()
		endforeach()
		if(major STREQUAL "" OR minor STREQUAL "")
			set(error "Could not read the Python version from '${inc}/patchlevel.h'. ${_td_installation_hint}")
		else()
			set(version "${major}.${minor}")
		endif()
	elseif(NOT error)
		if(WIN32)
			set(error "TouchDesigner's Python headers were not found in '${TD_INSTALLATION_PATH}/bin/include' or '${TD_3RDPARTY_PATH}/Python/Include'. ${_td_installation_hint}")
		else()
			set(error "TouchDesigner's Python.framework was not found in '${TD_FRAMEWORKS_PATH}'. ${_td_installation_hint}")
		endif()
	endif()

	# 2025 series builds keep pyconfig.h in Include/PC, as in a CPython source
	# tree, so that folder is needed on the include path as well. (2026 series
	# builds have it next to Python.h.)
	set(extra_incs "")
	if(version AND WIN32 AND inc STREQUAL "${TD_3RDPARTY_PATH}/Python/Include")
		if(EXISTS "${inc}/PC/pyconfig.h")
			list(APPEND extra_incs "${inc}/PC")
		else()
			set(error "TouchDesigner's Python pyconfig.h was not found in '${inc}/PC'. ${_td_installation_hint}")
			set(version "")
		endif()
	endif()

	if(version AND WIN32)
		# Windows' pyconfig.h is shared between the regular and free-threaded
		# builds, so which one TouchDesigner uses is told by the interpreter DLL
		# it ships (python314t.dll vs python314.dll), with the import library of
		# the same name. Never the _d debug variants: TouchDesigner's interpreter
		# is a release build.
		set(names "")
		if(EXISTS "${TD_INSTALLATION_PATH}/bin/python${major}${minor}t.dll")
			list(APPEND names "python${major}${minor}t")
		elseif(EXISTS "${TD_INSTALLATION_PATH}/bin/python${major}${minor}.dll")
			list(APPEND names "python${major}${minor}")
		else()
			# No DLL next to the import libraries: go by the libraries alone.
			list(APPEND names "python${major}${minor}t" "python${major}${minor}")
		endif()
		set(lib "")
		foreach(name IN LISTS names)
			if(EXISTS "${lib_dir}/${name}.lib")
				set(lib "${lib_dir}/${name}.lib")
				if(name MATCHES "t$")
					set(free_threaded TRUE)
				endif()
				break()
			endif()
		endforeach()
		if(lib)
			set(found TRUE)
			set(includes "${inc}" ${extra_incs})
			set(libs "${lib}")
			# pyconfig.h may auto-link a library with #pragma comment(lib); the
			# folder is on the link path so that resolves to the same library,
			# and Py_NO_LINK_LIB turns it off where supported (3.13+).
			set(lib_dirs "${lib_dir}")
			list(APPEND defs Py_NO_LINK_LIB)
			if(free_threaded)
				list(APPEND defs Py_GIL_DISABLED=1)
			endif()
		else()
			list(JOIN names ".lib or " expected)
			set(error "TouchDesigner's Python ${version} import library (${expected}.lib) was not found in '${lib_dir}'. ${_td_installation_hint}")
		endif()
	elseif(version AND APPLE)
		# No library to link: the plugin resolves Python's symbols from the
		# interpreter TouchDesigner has already loaded (-undefined dynamic_lookup,
		# added by td_add_custom_op), so no libpython path is baked in.
		# pyconfig.h is specific to the build here and defines Py_GIL_DISABLED
		# itself for a free-threaded Python.
		set(found TRUE)
		set(includes "${inc}")
		if(EXISTS "${inc}/pyconfig.h")
			file(STRINGS "${inc}/pyconfig.h" gil REGEX "^#define[ \t]+Py_GIL_DISABLED[ \t]+1")
			if(gil)
				set(free_threaded TRUE)
			endif()
		endif()
	endif()

	# The installation's Python must be the one these samples target; "t" marks
	# the free-threaded build, e.g. 3.14t.
	if(found)
		set(found_label "${version}")
		if(free_threaded)
			string(APPEND found_label "t")
		endif()
		set(expected_label "${TD_PYTHON_VERSION}")
		if(TD_PYTHON_FREE_THREADED)
			string(APPEND expected_label "t")
		endif()
		if(NOT found_label STREQUAL expected_label)
			set(found FALSE)
			set(error "TouchDesigner in '${TD_INSTALLATION_PATH}' ships Python ${found_label}, but these samples are looking for Python ${expected_label}, which ships with TouchDesigner ${TD_BUILD_SERIES} series builds. ${_td_installation_hint}")
		endif()
	endif()

	set(${prefix}_FOUND ${found} PARENT_SCOPE)
	set(${prefix}_INCLUDE_DIRS "${includes}" PARENT_SCOPE)
	set(${prefix}_LIBRARIES "${libs}" PARENT_SCOPE)
	set(${prefix}_LIBRARY_DIRS "${lib_dirs}" PARENT_SCOPE)
	set(${prefix}_DEFINITIONS "${defs}" PARENT_SCOPE)
	set(${prefix}_VERSION "${version}" PARENT_SCOPE)
	set(${prefix}_FREE_THREADED ${free_threaded} PARENT_SCOPE)
	set(${prefix}_ERROR "${error}" PARENT_SCOPE)
endfunction()

# ----------------------------------------------------------------------------
# Internal: locate the ONNX Runtime C API.
# ----------------------------------------------------------------------------
function(_td_find_onnxruntime prefix)
	set(hints "${ONNXRUNTIME_ROOT}")
	find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_c_api.h
		HINTS ${hints}
		PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
	find_path(ONNXRUNTIME_CONFIG_INCLUDE_DIR onnxruntime_run_options_config_keys.h
		HINTS ${hints}
		PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session)
	find_library(ONNXRUNTIME_LIBRARY onnxruntime
		HINTS ${hints}
		PATH_SUFFIXES lib lib/Release)
	if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_CONFIG_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
		set(${prefix}_FOUND TRUE PARENT_SCOPE)
		set(${prefix}_INCLUDE_DIRS "${ONNXRUNTIME_INCLUDE_DIR};${ONNXRUNTIME_CONFIG_INCLUDE_DIR}" PARENT_SCOPE)
		set(${prefix}_LIBRARIES "${ONNXRUNTIME_LIBRARY}" PARENT_SCOPE)
	else()
		set(${prefix}_FOUND FALSE PARENT_SCOPE)
		set(${prefix}_ERROR "ONNX Runtime not found. Set ONNXRUNTIME_ROOT to the extracted onnxruntime package (currently '${ONNXRUNTIME_ROOT}')." PARENT_SCOPE)
	endif()
endfunction()

# ----------------------------------------------------------------------------
# td_add_custom_op(<target> ...)
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# td_skip_custom_op(<target> <reason>)
# ----------------------------------------------------------------------------
function(td_skip_custom_op target reason)
	if(TD_CUSTOM_OPS_ROOT_BUILD)
		# Building all samples: skip this one, keep going with the rest.
		message(WARNING "Skipping ${target}: ${reason}")
		set_property(GLOBAL APPEND PROPERTY TD_SKIPPED_SAMPLES "${target}")
	else()
		message(FATAL_ERROR "Cannot build ${target}: ${reason}")
	endif()
endfunction()

# ----------------------------------------------------------------------------
# td_add_custom_op(<target> ...)
# ----------------------------------------------------------------------------
function(td_add_custom_op target)
	cmake_parse_arguments(PARSE_ARGV 1 ARG
		"CUDA;PYTHON;ONNXRUNTIME;WINDOWS_ONLY"
		"CXX_STANDARD;FAMILY"
		"SOURCES;INCLUDE_DIRECTORIES;OPENCV_COMPONENTS;LINK_LIBRARIES")

	if(NOT ARG_SOURCES)
		message(FATAL_ERROR "td_add_custom_op(${target}): SOURCES is required")
	endif()
	if(NOT ARG_CXX_STANDARD)
		set(ARG_CXX_STANDARD 17)
	endif()

	# Family (CHOP/DAT/SOP/TOP...) is used for IDE folders; by default it is the
	# name of the parent folder (e.g. CHOP/BasicFilterCHOP).
	if(ARG_FAMILY)
		set(family "${ARG_FAMILY}")
	else()
		get_filename_component(family_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
		get_filename_component(family "${family_dir}" NAME)
	endif()

	# ---- Dependency checks --------------------------------------------------
	set(missing "")

	if(ARG_WINDOWS_ONLY AND NOT WIN32)
		list(APPEND missing "this sample only builds on Windows")
	endif()

	# CUDA samples turned off on purpose: in the all-samples build that's not a
	# problem to warn about. (Configuring the sample on its own with the option
	# off still stops, via the CUDA check below.)
	if(ARG_CUDA AND NOT missing AND NOT TD_BUILD_CUDA_SAMPLES AND TD_CUSTOM_OPS_ROOT_BUILD)
		message(STATUS "Skipping ${target}: the CUDA samples are turned off (TD_BUILD_CUDA_SAMPLES is OFF)")
		set_property(GLOBAL APPEND PROPERTY TD_SKIPPED_SAMPLES "${target}")
		return()
	endif()

	if(ARG_CUDA AND NOT missing)
		if(TD_CUDA_ERROR)
			list(APPEND missing "${TD_CUDA_ERROR}")
		elseif(NOT TD_CUDA_AVAILABLE OR NOT TARGET CUDA::cudart)
			list(APPEND missing "the CUDA Toolkit was not found (install it, or set CMAKE_CUDA_COMPILER to nvcc)")
		endif()
	endif()

	if(ARG_OPENCV_COMPONENTS AND NOT missing)
		_td_find_opencv(cv ${ARG_OPENCV_COMPONENTS})
		if(NOT cv_FOUND)
			list(APPEND missing "OpenCV: ${cv_ERROR}")
		endif()
	endif()

	if(ARG_PYTHON AND NOT missing)
		_td_find_python(py)
		if(NOT py_FOUND)
			list(APPEND missing "${py_ERROR}")
		endif()
	endif()

	if(ARG_ONNXRUNTIME AND NOT missing)
		_td_find_onnxruntime(ort)
		if(NOT ort_FOUND)
			list(APPEND missing "${ort_ERROR}")
		endif()
	endif()

	if(missing)
		list(JOIN missing "; " reason)
		td_skip_custom_op(${target} "${reason}")
		return()
	endif()

	# Report what the sample is built against.
	set(notes "")
	if(ARG_OPENCV_COMPONENTS)
		if(cv_BUNDLED)
			list(APPEND notes "OpenCV ${cv_VERSION} from TouchDesigner")
		else()
			list(APPEND notes "OpenCV ${cv_VERSION}")
		endif()
	endif()
	if(ARG_PYTHON)
		if(py_FREE_THREADED)
			list(APPEND notes "Python ${py_VERSION}t from TouchDesigner")
		else()
			list(APPEND notes "Python ${py_VERSION} from TouchDesigner")
		endif()
	endif()
	if(notes)
		list(JOIN notes ", " notes)
		message(STATUS "Adding ${family} sample: ${target} (${notes})")
	else()
		message(STATUS "Adding ${family} sample: ${target}")
	endif()

	# ---- Target -------------------------------------------------------------
	add_library(${target} MODULE ${ARG_SOURCES})

	target_compile_features(${target} PRIVATE cxx_std_${ARG_CXX_STANDARD})
	set_target_properties(${target} PROPERTIES
		CXX_EXTENSIONS OFF
		PREFIX ""
		FOLDER "${family}")

	if(ARG_INCLUDE_DIRECTORIES)
		target_include_directories(${target} PRIVATE ${ARG_INCLUDE_DIRECTORIES})
	endif()

	if(TD_PLUGIN_OUTPUT_DIR)
		set(out_dir "${TD_PLUGIN_OUTPUT_DIR}")
	else()
		set(out_dir "${_td_plugin_output_default}")
	endif()
	# The $<1:...> generator expression stops multi-config generators (Visual
	# Studio, Xcode) from appending a /Debug or /Release sub folder.
	set_target_properties(${target} PROPERTIES
		LIBRARY_OUTPUT_DIRECTORY "$<1:${out_dir}>"
		RUNTIME_OUTPUT_DIRECTORY "$<1:${out_dir}>")

	if(APPLE)
		# TouchDesigner loads custom operators on macOS as .plugin bundles.
		# CMake writes each bundle's Info.plist from a template; its default one
		# is meant for applications, so use our own (loadable bundle, versioned).
		# The bundle version comes from the sample's project(... VERSION x.y.z).
		if(PROJECT_VERSION)
			set(bundle_version "${PROJECT_VERSION}")
		else()
			set(bundle_version "1.0.0")
		endif()
		set_target_properties(${target} PROPERTIES
			BUNDLE TRUE
			BUNDLE_EXTENSION "plugin"
			MACOSX_BUNDLE_INFO_PLIST "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/TouchDesignerPluginInfo.plist.in"
			MACOSX_BUNDLE_GUI_IDENTIFIER "${TD_BUNDLE_IDENTIFIER_PREFIX}.${target}"
			MACOSX_BUNDLE_BUNDLE_NAME "${target}"
			MACOSX_BUNDLE_SHORT_VERSION_STRING "${bundle_version}"
			MACOSX_BUNDLE_BUNDLE_VERSION "${bundle_version}"
			XCODE_ATTRIBUTE_PRODUCT_BUNDLE_IDENTIFIER "${TD_BUNDLE_IDENTIFIER_PREFIX}.${target}"
			# Product > Archive only copies targets that install into the
			# archive, and puts them at Products/$(INSTALL_PATH); bundle targets
			# default to SKIP_INSTALL=YES, which leaves the archive empty.
			XCODE_ATTRIBUTE_SKIP_INSTALL NO
			XCODE_ATTRIBUTE_INSTALL_PATH "/")

		# Newer TouchDesigner headers `#import <Metal/Metal.h>` and use
		# id<MTL...> types on macOS, so the sources must be Objective-C++.
		set(uses_metal FALSE)
		foreach(src IN LISTS ARG_SOURCES)
			get_filename_component(src_name "${src}" NAME)
			if(src_name STREQUAL "CPlusPlus_Common.h")
				get_filename_component(src_path "${src}" ABSOLUTE)
				file(STRINGS "${src_path}" metal_import REGEX "#[ \t]*import[ \t]*<Metal/Metal\\.h>")
				if(metal_import)
					set(uses_metal TRUE)
				endif()
			endif()
		endforeach()
		if(uses_metal)
			foreach(src IN LISTS ARG_SOURCES)
				if(src MATCHES "\\.(cpp|cc|cxx)$")
					set_source_files_properties("${src}" PROPERTIES LANGUAGE OBJCXX)
				endif()
			endforeach()
			set_target_properties(${target} PROPERTIES
				OBJCXX_STANDARD ${ARG_CXX_STANDARD}
				OBJCXX_STANDARD_REQUIRED ON
				OBJCXX_EXTENSIONS OFF)
			target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
			target_link_libraries(${target} PRIVATE "-framework Foundation" "-framework Metal")
		endif()
	endif()

	if(MSVC)
		target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/MP /W3>)
		# The samples use the Unicode Win32 APIs, as their original Visual
		# Studio projects did (CharacterSet = Unicode). Without this, CMake
		# 4.2+ defines _MBCS for MSVC targets (policy CMP0204).
		target_compile_definitions(${target} PRIVATE UNICODE _UNICODE)
	endif()

	# ---- Dependencies -------------------------------------------------------
	if(ARG_OPENCV_COMPONENTS)
		target_include_directories(${target} PRIVATE ${cv_INCLUDE_DIRS})
		target_link_libraries(${target} PRIVATE ${cv_LIBRARIES})
		if(MSVC AND cv_BUNDLED)
			# TouchDesigner ships Release builds of OpenCV. Use the release CRT in
			# every configuration so Debug builds can link against it too.
			set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
		endif()
		if(APPLE AND cv_BUNDLED)
			# The bundled dylibs are @rpath/libopencv_*.dylib; resolve them from
			# TouchDesigner's Frameworks folder when the plugin is loaded.
			# Link with the final rpath so `cmake --install` doesn't rewrite it
			# with install_name_tool, which would break the arm64 code signature.
			set_target_properties(${target} PROPERTIES
				BUILD_WITH_INSTALL_RPATH TRUE
				INSTALL_RPATH "@executable_path/../Frameworks")
		endif()
	endif()

	if(ARG_CUDA)
		# The CUDA runtime is linked explicitly through CUDA::cudart (cudart.lib),
		# the dynamic runtime. From CUDA 13, nvcc on Windows names that mode
		# "hybrid" (cudart.lib now binds to the runtime in the display driver,
		# nvcudart_hybrid64.dll) and warns that "shared" is deprecated. CMake
		# has no value for "hybrid", so nvcc is told -cudart=none and the same
		# library is linked by the line below; CUDA 12 keeps "shared".
		if(TD_CUDA_MAJOR_VERSION VERSION_GREATER_EQUAL 13)
			set(cuda_runtime None)
		else()
			set(cuda_runtime Shared)
		endif()
		set_target_properties(${target} PROPERTIES
			CUDA_STANDARD 17
			CUDA_RUNTIME_LIBRARY ${cuda_runtime})
		target_link_libraries(${target} PRIVATE CUDA::cudart)
		if(TD_CUDA_ALLOW_UNSUPPORTED_COMPILER)
			target_compile_definitions(${target} PRIVATE _ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
		endif()
	endif()

	if(ARG_PYTHON)
		target_include_directories(${target} PRIVATE ${py_INCLUDE_DIRS})
		target_compile_definitions(${target} PRIVATE ${py_DEFINITIONS})
		if(py_LIBRARIES)
			target_link_libraries(${target} PRIVATE ${py_LIBRARIES})
		endif()
		if(py_LIBRARY_DIRS)
			target_link_directories(${target} PRIVATE ${py_LIBRARY_DIRS})
		endif()
		if(MSVC)
			# TouchDesigner's Python is a release build. The debug CRT (/MDd)
			# defines _DEBUG, which makes pyconfig.h define Py_DEBUG (a different
			# object layout) and link the _d library, so use the release CRT in
			# every configuration.
			set_target_properties(${target} PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
		endif()
		if(APPLE)
			# Resolve Python's symbols from the interpreter TouchDesigner has
			# already loaded, rather than linking a libpython.
			target_link_options(${target} PRIVATE "LINKER:-undefined,dynamic_lookup")
		endif()
	endif()

	if(ARG_ONNXRUNTIME)
		target_include_directories(${target} PRIVATE ${ort_INCLUDE_DIRS})
		target_link_libraries(${target} PRIVATE ${ort_LIBRARIES})
	endif()

	if(ARG_LINK_LIBRARIES)
		target_link_libraries(${target} PRIVATE ${ARG_LINK_LIBRARIES})
	endif()

	# ---- Debugging: launch TouchDesigner with the text console ---------------
	if(_td_debug_executable)
		set_target_properties(${target} PROPERTIES
			VS_DEBUGGER_COMMAND "${_td_debug_executable}"
			VS_DEBUGGER_ENVIRONMENT "TOUCH_TEXT_CONSOLE=1"
			VS_DEBUGGER_WORKING_DIRECTORY "${out_dir}"
			XCODE_SCHEME_EXECUTABLE "${_td_debug_executable}"
			XCODE_SCHEME_ENVIRONMENT "TOUCH_TEXT_CONSOLE=1")
	endif()

	# ---- Install: copy the plugin into TouchDesigner's Plugins folder ---------
	# MODULE libraries (.dll, and .plugin bundles on macOS) use LIBRARY.
	install(TARGETS ${target}
		LIBRARY DESTINATION .
		RUNTIME DESTINATION .)

	# When this sample is the top-level project, make it the VS startup project.
	if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
		set_property(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}" PROPERTY VS_STARTUP_PROJECT ${target})
	endif()
endfunction()
