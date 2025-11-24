cmake_minimum_required(VERSION 3.14)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

# Add your test executable
add_executable(my_tests my_tests.cpp)

# Link your test executable to gtest and your project's library
target_link_libraries(my_tests PRIVATE gtest_main mmWaveLib)

# Discover tests using gtest_discover_tests
include(GoogleTest)
add_executable(area_scanner_tests area_scanner_tests.cpp)

# Link your test executable to gtest and your project's library
target_link_libraries(area_scanner_tests PRIVATE gtest_main mmWaveLib)

# Discover tests using gtest_discover_tests
include(GoogleTest)
gtest_discover_tests(area_scanner_tests)