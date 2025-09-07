cmake_minimum_required(VERSION 3.18)
project(lbm_fortran LANGUAGES Fortran)

# Options
set(CMAKE_BUILD_TYPE Release)
add_executable(lbm lbm_cylinder.f90)

# Reasonable flags
if (CMAKE_Fortran_COMPILER_ID MATCHES "GNU|Intel")
  target_compile_options(lbm PRIVATE -O3 -ffast-math)
endif()
