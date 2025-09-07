cmake_minimum_required(VERSION 3.18)
project(nbody_c LANGUAGES C)

set(CMAKE_BUILD_TYPE Release)

add_executable(nbody
  main.c
  nbody.c
)

target_compile_features(nbody PRIVATE c_std_11)

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(nbody PRIVATE -O3 -ffast-math -march=native)
endif()
