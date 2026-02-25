cmake_minimum_required(VERSION 3.14)
project(tile_shape_inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# Options
# ============================================================================

option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_EXAMPLES "Build example programs" ON)

# ============================================================================
# Sources
# ============================================================================

set(TILE_SHAPE_SOURCES
    tile_shape_inference_pass.cpp
    op_builders.cpp
)

set(TILE_SHAPE_HEADERS
    tile_shape_inference_pass.h
    op_builders.h
)

# ============================================================================
# Library
# ============================================================================

add_library(tile_shape_inference STATIC
    ${TILE_SHAPE_SOURCES}
    ${TILE_SHAPE_HEADERS}
)

target_include_directories(tile_shape_inference
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
        $<INSTALL_INTERFACE:include>
)

# ============================================================================
# Example Program
# ============================================================================

if(BUILD_EXAMPLES)
    add_executable(tile_shape_example example/main.cpp)
    target_link_libraries(tile_shape_example PRIVATE tile_shape_inference)
endif()

# ============================================================================
# Tests
# ============================================================================

if(BUILD_TESTS)
    enable_testing()

    add_executable(tile_shape_test tests/test_pass.cpp)
    target_link_libraries(tile_shape_test PRIVATE tile_shape_inference)

    add_test(NAME TileShapeInferenceTest COMMAND tile_shape_test)
endif()

# ============================================================================
# Install
# ============================================================================

install(TARGETS tile_shape_inference
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

install(FILES ${TILE_SHAPE_HEADERS}
    DESTINATION include/tile_shape
)
