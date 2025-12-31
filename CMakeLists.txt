cmake_minimum_required(VERSION 3.15)
project(
    FACS
    VERSION 0.2
    HOMEPAGE_URL "https://github.com/12ff54e/FACS.git"
    LANGUAGES CXX)

include(CTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set default build type to debug.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
message(STATUS "Build type = " ${CMAKE_BUILD_TYPE})

# Check compile flags availability
include(CheckCXXCompilerFlag)

set(asan_flag -fsanitize=address)
set(CMAKE_REQUIRED_LIBRARIES ${asan_flag}) # This is also a linker flag
check_cxx_compiler_flag(${asan_flag} compiler_support_asan)
if(${compiler_support_asan})
    add_compile_options($<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${asan_flag}>)
    add_link_options($<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:${asan_flag}>)
endif()

# Add options according to compiler.
if(MSVC)
    set(CMAKE_CXX_FLAGS_INIT "/W4 /EHsc")
else()
    if(CMAKE_CXX_COMPILER_ID STREQUAL Clang)
        set(CMAKE_CXX_FLAGS_INIT "-Weverything -Wno-c++98-compat -Wno-c++98-compat-pedantic -Wno-padded -Wno-exit-time-destructors")
    else()
        set(CMAKE_CXX_FLAGS_INIT "-Wall -Wextra")
    endif()
endif()

add_compile_definitions(INTP_CELL_LAYOUT)

add_executable(facs src/main.cpp src/gFileRawData.cpp)
target_include_directories(facs PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

add_executable(test_two_dof src/test_two_dof.cpp src/gFileRawData.cpp)
target_include_directories(test_two_dof PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)

add_executable(test_equilibrium_list src/test_equilibrium_list.cpp src/gFileRawData.cpp)
target_include_directories(test_equilibrium_list PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)

add_executable(test_resonance_continuum src/test_resonance_continuum.cpp src/gFileRawData.cpp)
target_include_directories(test_resonance_continuum PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)

add_executable(test_single_surface src/test_single_surface.cpp src/gFileRawData.cpp)
target_include_directories(test_single_surface PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)

add_executable(test_interpolation src/test_interpolation.cpp src/gFileRawData.cpp)
target_include_directories(test_interpolation PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>)
