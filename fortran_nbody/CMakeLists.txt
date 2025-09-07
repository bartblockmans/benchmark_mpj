cmake_minimum_required(VERSION 3.10)
project(nbody LANGUAGES Fortran)

# Set Fortran standard
set(CMAKE_Fortran_STANDARD 2008)
set(CMAKE_Fortran_STANDARD_REQUIRED ON)

# Set compiler flags for optimization - more conservative but effective
if(CMAKE_Fortran_COMPILER_ID MATCHES "GNU")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -march=native -ffast-math -funroll-loops -ftree-vectorize -fvect-cost-model=unlimited -fno-protect-parens")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -fcheck=all -fbacktrace")
    set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-O2 -g -fcheck=all -fbacktrace")
    # Fix linking issues on Windows with MinGW
    set(CMAKE_Fortran_LINK_FLAGS "-static-libgcc -static-libgfortran")
    set(CMAKE_EXE_LINKER_FLAGS "-static-libgcc -static-libgfortran")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "Intel")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -xHost -no-prec-div -qopt-report=5 -qopt-report-phase=vec,loop")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -check all -traceback")
    set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-O2 -g -check all -traceback")
elseif(CMAKE_Fortran_COMPILER_ID MATCHES "PGI")
    set(CMAKE_Fortran_FLAGS_RELEASE "-O3 -fast -Munroll -Mvect=simd -Minfo=all")
    set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0 -Mbounds -Mchkptr")
    set(CMAKE_Fortran_FLAGS_RELWITHDEBINFO "-O2 -g -Mbounds -Mchkptr")
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

# Create output directory for images
file(MAKE_DIRECTORY ${CMAKE_BINARY_DIR}/images)

# Add executable with both source files
add_executable(nbody
    json_reader.f90
    nbody.f90
)

# Set output directory and executable name
set_target_properties(nbody PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    OUTPUT_NAME nbody
    LINK_FLAGS "-static-libgcc -static-libgfortran"
)

# Copy initial conditions file if it exists
if(EXISTS "${CMAKE_SOURCE_DIR}/../nbody_ic_galaxy_spiral_N4000.json")
    configure_file(
        ${CMAKE_SOURCE_DIR}/../nbody_ic_galaxy_spiral_N4000.json
        ${CMAKE_BINARY_DIR}/nbody_ic_galaxy_spiral_N4000.json
        COPYONLY
    )
    message(STATUS "Copied initial conditions file to build directory")
else()
    message(WARNING "Initial conditions file not found: ../nbody_ic_galaxy_spiral_N4000.json")
endif()

# Print configuration information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Fortran compiler: ${CMAKE_Fortran_COMPILER}")
message(STATUS "Fortran compiler ID: ${CMAKE_Fortran_COMPILER_ID}")
message(STATUS "Fortran flags: ${CMAKE_Fortran_FLAGS}")
message(STATUS "Fortran linker: ${CMAKE_Fortran_LINKER_LAUNCHER}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Add install target
install(TARGETS nbody
    RUNTIME DESTINATION bin
)

# Add custom target for running the simulation
add_custom_target(run_nbody
    COMMAND ${CMAKE_BINARY_DIR}/nbody.exe
    DEPENDS nbody
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running N-body simulation"
)

# Add custom target for running with default parameters
add_custom_target(run_nbody_default
    COMMAND ${CMAKE_BINARY_DIR}/nbody.exe 1.0 100 1 1
    DEPENDS nbody
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running N-body simulation with default parameters"
)

# Add custom target for running without images
add_custom_target(run_nbody_no_images
    COMMAND ${CMAKE_BINARY_DIR}/nbody.exe 1.0 100 0 1
    DEPENDS nbody
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running N-body simulation without image generation"
)
