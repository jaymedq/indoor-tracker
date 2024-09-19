# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Create a list of source files
set(SOURCES
    frames/AreaScannerFrame.cpp
    frames/DemoFrame.cpp
    main.cpp
)

# Create a list of header files
set(HEADERS
    frames/AreaScannerFrame.hpp
    frames/DemoFrame.hpp
    frames/IFrame.hpp
)

# Platform-specific options
if(CMAKE_SYSTEM_NAME STREQUAL "Windows" OR CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # Set standard libraries and flags for Windows and Raspberry Pi (Linux)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_executable(IWR6843Interface ${SOURCES} ${HEADERS})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "ESP32")
    # Include ESP-IDF component
    include($ENV{IDF_PATH}/tools/cmake/project.cmake)

    # Use standard C++ library with ESP-IDF (if available)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    # Add the executable for ESP32
    add_executable(IWR6843Interface ${SOURCES})

    # Check for required components or alternative libraries
    # For example: using freertos or esp-idf components
    target_link_libraries(IWR6843Interface PRIVATE "-u __cxa_guard_dummy -lstdc++")
endif()
