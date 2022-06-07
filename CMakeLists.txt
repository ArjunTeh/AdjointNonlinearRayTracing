# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.9)
cmake_policy(VERSION 3.9)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project ("Eikonal-Enoki" LANGUAGES CXX CUDA)
include(CheckCXXCompilerFlag)

set(CMAKE_CXX_STANDARD 17)
if( WIN32 )
  add_definitions(-D_USE_MATH_DEFINES -D_CRT_SECURE_NO_WARNINGS -DNDEBUG)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -Wall -fPIC")
endif()

if (POLICY CMP0056)
  cmake_policy(SET CMP0056 NEW) # try_compile: pass linker flags to compiler
endif()

macro(CHECK_CXX_COMPILER_AND_LINKER_FLAGS _RESULT _CXX_FLAGS _LINKER_FLAGS)
  set(CMAKE_REQUIRED_FLAGS ${_CXX_FLAGS})
  set(CMAKE_REQUIRED_LIBRARIES ${_LINKER_FLAGS})
  set(CMAKE_REQUIRED_QUIET TRUE)
  set(CMAKE_REQUIRED_FLAGS "")
  set(CMAKE_REQUIRED_LIBRARIES "")
endmacro()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND NOT CMAKE_CXX_FLAGS MATCHES "-stdlib=libc\\+\\+")
  CHECK_CXX_COMPILER_AND_LINKER_FLAGS(HAS_LIBCPP "-stdlib=libc++" "-stdlib=libc++")
  if (HAS_LIBCPP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -D_LIBCPP_VERSION")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -stdlib=libc++")
    message(STATUS "drrt: using libc++.")
  endif()
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
# if ( WIN32 )
#   set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/$<0:>)
# endif()

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
  # foreach( OUTPUTCONFIG ${CMAKE_CONFIGURATION_TYPES} )
  #   string( TOUPPER ${OUTPUTCONFIG} OUTPUTCONFIG )
  #   set( CMAKE_RUNTIME_OUTPUT_DIRECTORY_${OUTPUTCONFIG} $<0:>)
  #   set( CMAKE_LIBRARY_OUTPUT_DIRECTORY_${OUTPUTCONFIG} $<0:>)
  #   set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${OUTPUTCONFIG} $<0:>)
  # endforeach( OUTPUTCONFIG CMAKE_CONFIGURATION_TYPES )
endif()

# Include sub-projects.
add_subdirectory(ext)

enoki_set_compile_flags()
enoki_set_native_flags()

include_directories(
	include/
  ${ENOKI_INCLUDE_DIRS}
  ext/enoki/ext/pybind11/include
)

add_library(
	eikonal-tracer
	include/volume.h
	src/volume.cpp
    src/cylinder_volume.cpp
	include/tracer.h
    src/tracer.cpp
)

target_link_libraries(eikonal-tracer PUBLIC enoki-cuda enoki-autodiff cuda)
# set_target_properties(eikonal-tracer PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/drrt)

# add_custom_command(
#   OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/drrt/__init__.py
#   COMMAND ${CMAKE_COMMAND} -E copy
#   ${CMAKE_CURRENT_SOURCE_DIR}/src/__init__.py
#   ${CMAKE_CURRENT_BINARY_DIR}/drrt/__init__.py
#   )
# add_custom_target(
#   drrt-python-init
#   ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/drrt/__init__.py
#   )


pybind11_add_module(drrt src/drrt.cpp)
target_link_libraries(drrt PUBLIC eikonal-tracer cuda enoki-cuda enoki-autodiff)
# target_compile_options(drrt PRIVATE /wd4251)
# set_target_properties(drrt PROPERTIES
#   LIBRARY_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/drrt
  # RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/drrt
  # ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/drrt
  # )

add_executable(run src/test.cpp)
target_link_libraries(run PUBLIC eikonal-tracer cuda enoki-autodiff enoki-cuda)
