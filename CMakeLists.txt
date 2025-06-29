# CMake minimum version
cmake_minimum_required(VERSION 3.10)

# Project name
project(RGDIC)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenMP
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(your_target PUBLIC OpenMP::OpenMP_CXX)
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(Eigen3)
include_directories(fftw3/include)

# Add executable
add_executable(rgdic_demo
    main.cpp
    rgdic.cpp
    icgn_optimizer.cpp
    neighbor_utils.cpp
)

# Link libraries
target_link_libraries(rgdic_demo ${OpenCV_LIBS})

# Add custom target for g++ compilation
add_custom_target(compile_with_gpp
    COMMAND g++ -fdiagnostics-color=always -fopenmp -g main.cpp rgdic.cpp icgn_optimizer.cpp neighbor_utils.cpp 
            -I Eigen3 -I fftw3/include -I ${OpenCV_INCLUDE_DIRS}
            -L fftw3/lib -L ${OpenCV_LIB_DIR}
            -l fftw3 -l fftw3f -l fftw3l ${OpenCV_LIBS}
            -o rgdic_demo.exe
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Compiling with g++ directly"
)
