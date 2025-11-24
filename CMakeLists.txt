# project-root/CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(SensorFusionProject CXX)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
enable_testing()

# Define a variable for the project's main include directory
set(PROJECT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")

# --- THIRD-PARTY: Serial Library Definition ---
# Define sources for the 'serial' library
set(SERIAL_BASE_SRCS
    third-party/src/serial.cc
)

# Platform-specific logic for serial library
if(WIN32)
    list(APPEND SERIAL_BASE_SRCS third-party/src/impl/win.cc)
    set(SERIAL_LIBS "SetupAPI")
elseif(UNIX AND NOT APPLE)
    list(APPEND SERIAL_BASE_SRCS third-party/src/impl/unix.cc)
    set(SERIAL_LIBS "rt")
elseif(APPLE)
    list(APPEND SERIAL_BASE_SRCS third-party/src/impl/unix.cc)
    # No specific libs needed for macOS
endif()

# Add the 'serial' library
add_library(serial STATIC ${SERIAL_BASE_SRCS})

# Set include directories for consumers of the serial library
target_include_directories(serial PUBLIC 
    third-party/include
)

# Link platform-specific libraries
if(SERIAL_LIBS)
    target_link_libraries(serial PRIVATE ${SERIAL_LIBS})
endif()

# Add the 'src' directory, which will handle all sub-library definitions
add_subdirectory(src)

# Add the 'tests' directory
add_subdirectory(tests)