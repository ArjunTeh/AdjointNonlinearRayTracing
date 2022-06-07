## Helper file for setting the appropriate variables

# Add pybind library
# set(Python_ROOT_DIR "C:/Users/chind/.conda/envs/eikonal/python.exe")
# add_subdirectory(pybind11)

add_subdirectory(enoki)

enoki_set_compile_flags()
enoki_set_native_flags()
# set_property(TARGET enoki-cuda PROPERTY CUDA_ARCHITECTURES OFF)
get_directory_property(ENOKI_COMPILE_OPTIONS     COMPILE_OPTIONS)
get_directory_property(ENOKI_COMPILE_DEFINITIONS COMPILE_DEFINITIONS)
set_property(DIRECTORY .. PROPERTY COMPILE_OPTIONS     ${ENOKI_COMPILE_OPTIONS})
set_property(DIRECTORY .. PROPERTY COMPILE_DEFINITIONS ${ENOKI_COMPILE_DEFINITIONS})
set(ENOKI_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/enoki/include PARENT_SCOPE)
set(CMAKE_CXX_STANDARD_LIBRARIES ${CMAKE_CXX_STANDARD_LIBRARIES} PARENT_SCOPE)
set(CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS} PARENT_SCOPE)
set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} PARENT_SCOPE)
