cmake_minimum_required(VERSION 3.28)

# This fork targets AMD gfx906 (Instinct MI50/MI60) through HIP. Set and
# validate the architecture before HIP compiler detection so CMake never
# injects a generic host-toolchain default or accepts an unsupported target.
if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES gfx906 CACHE STRING "HIP architectures to build")
endif()
if(NOT CMAKE_HIP_ARCHITECTURES STREQUAL "gfx906")
  message(FATAL_ERROR
    "This NInfer fork supports only CMAKE_HIP_ARCHITECTURES=gfx906; got "
    "'${CMAKE_HIP_ARCHITECTURES}'")
endif()

project(ninfer LANGUAGES C CXX HIP)

# --- Global settings ---------------------------------------------------------
option(NINFER_BUILD_APPS "Build the NInfer CLI and server" ON)
option(BUILD_TESTING "Build NInfer tests" OFF)
option(NINFER_BUILD_BENCHMARKS "Build NInfer benchmarks" OFF)
option(NINFER_ENABLE_MEDIA "Build FFmpeg-based media decode (vision inputs)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

# gfx906 down-port compatibility mode: PDL/cp.async/tensor-core code paths are
# downgraded or stubbed behind this macro (pattern: ninfer-3090's
# NINFER_SM8X_COMPAT). It applies to every target in the tree.
add_compile_definitions(NINFER_GFX906_COMPAT=1)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)        # clangd / IDE tooling
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS cuda_link=1)
  set(CMAKE_JOB_POOL_LINK cuda_link)
endif()

if(CMAKE_HIP_COMPILER_VERSION VERSION_LESS 17)
  message(FATAL_ERROR
    "NInfer gfx906 requires a ROCm 6.x clang (>= 17); found HIP compiler "
    "${CMAKE_HIP_COMPILER_VERSION}")
endif()

set(NINFER_BUILD_MEDIA_ACQUIRE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_MEDIA_ACQUIRE ON)
endif()

set(NINFER_BUILD_PROMPT_INPUT OFF)
if(NINFER_BUILD_APPS)
  set(NINFER_BUILD_PROMPT_INPUT ON)
endif()

set(NINFER_BUILD_SERVE OFF)
if(NINFER_BUILD_APPS OR BUILD_TESTING)
  set(NINFER_BUILD_SERVE ON)
endif()

find_package(hip REQUIRED CONFIG)
if(NINFER_ENABLE_MEDIA)
  find_package(PkgConfig REQUIRED)
  pkg_check_modules(FFMPEG REQUIRED IMPORTED_TARGET
    libavformat>=60 libavcodec>=60 libavutil>=58 libswscale>=7)
  if(NINFER_BUILD_MEDIA_ACQUIRE)
    pkg_check_modules(LIBCURL REQUIRED IMPORTED_TARGET libcurl>=7.85)
  endif()
endif()
find_package(Threads REQUIRED)

# --- Subdirectories ----------------------------------------------------------
add_subdirectory(src)
if(NINFER_BUILD_APPS)
  add_subdirectory(apps)
endif()
if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
if(NINFER_BUILD_BENCHMARKS)
  add_subdirectory(bench)
endif()
