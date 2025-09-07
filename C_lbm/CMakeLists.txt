cmake_minimum_required(VERSION 3.18)
project(lbm_c LANGUAGES C)

set(CMAKE_BUILD_TYPE Release)

add_executable(lbm
  main.c
  lbm.c
  viz_ppm.c
  config.c
)

target_compile_features(lbm PRIVATE c_std_11)

if (CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(lbm PRIVATE -O3 -ffast-math -march=native)
endif()