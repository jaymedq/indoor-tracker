cmake_minimum_required(VERSION 3.10)
project(IWR6843Interface)

set(CMAKE_CXX_STANDARD 14)

# Define the shared library
add_library(IWR6843Interface SHARED
    src/frames/AreaScannerFrame.cpp
    src/frames/DemoFrame.cpp
    src/parser.cpp  # Ensure this file is included
)

# Specify include directories for the shared library
target_include_directories(IWR6843Interface PUBLIC src/frames)
target_include_directories(IWR6843Interface PUBLIC src)

# Define the executable
add_executable(IWR6843TestApp src/main.cpp)

# Link the executable with the shared library
target_link_libraries(IWR6843TestApp IWR6843Interface)
