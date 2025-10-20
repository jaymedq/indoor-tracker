cmake_minimum_required(VERSION 3.14)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/release-1.14.0.zip
  URL_HASH SHA256=10f7690623253b632003c73335359a164c30113c1292193b2a16f80907d03a08
)

FetchContent_MakeAvailable(googletest)

# Add your test executable
add_executable(my_tests tests/my_test.cpp)

# Link your test executable to gtest and your project's library
target_link_libraries(my_tests PRIVATE gtest_main my_library)

# Discover tests using gtest_discover_tests
include(GoogleTest)
gtest_discover_tests(my_tests)