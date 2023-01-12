cmake_minimum_required(VERSION 3.20)
project(causal)

set(CMAKE_CXX_STANDARD 20)
add_compile_options("-mavx512f" "-march=native" -Wall)

include(FetchContent)
FetchContent_Declare(
        argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
)
FetchContent_MakeAvailable(argparse)

option(ASSEMBLY "Emit assembly output")
if(${ASSEMBLY})
    add_compile_options("-S")
endif()

add_executable(test-avx512 test-avx512.cpp)
add_executable(test-avx512-compiler test-avx512.cpp)
target_compile_definitions(test-avx512-compiler PUBLIC COMPILER=1)

macro(causal_executable NAME POLICY)
    set(TARGET causal-${NAME})
    add_executable(${TARGET} main.cpp)
    target_link_libraries(${TARGET} argparse)
    target_compile_definitions(${TARGET} PUBLIC POLICY=${POLICY})
    if (${ARGC} GREATER 2)
        target_compile_options(${TARGET} PUBLIC ${ARGV2})
    endif ()
    if (${ARGC} GREATER 3)
        target_link_options(${TARGET} PUBLIC ${ARGV3})
    endif ()
endmacro()

causal_executable(serial SERIAL "-Rpass=vectorize")
causal_executable(serial-fast SERIAL_FAST "")
causal_executable(serial-vec SERIAL_VEC "")
causal_executable(cilk PARALLEL "-fopencilk" "-fopencilk")
causal_executable(omp PARALLEL_OMP "-fopenmp" "-lgomp")

# tests
