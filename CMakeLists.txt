cmake_minimum_required(VERSION 3.10)
project(IWR6843Interface)

set(CMAKE_CXX_STANDARD 14)

enable_testing()

# Define the shared library
add_library(IWR6843Interface SHARED
    src/frames/AreaScannerFrame.cpp
    src/frames/DemoFrame.cpp
    src/frames/PeopleTrackingFrame.cpp
    src/parser.cpp  # Ensure this file is included
)

# Specify include directories for the shared library
target_include_directories(IWR6843Interface PUBLIC src/frames)
target_include_directories(IWR6843Interface PUBLIC src)

# Serial library
target_include_directories(IWR6843Interface PUBLIC third-party/include)
include_directories(third-party/include)
## Sources
set(serial_SRCS
    third-party/src/serial.cc
    third-party/include/serial/serial.h
    third-party/include/serial/v8stdint.h
)
if(APPLE)
    # If OSX
    list(APPEND serial_SRCS third-party/src/impl/unix.cc)
    list(APPEND serial_SRCS third-party/src/impl/list_ports/list_ports_osx.cc)
elseif(UNIX)
    # If unix
    list(APPEND serial_SRCS third-party/src/impl/unix.cc)
    list(APPEND serial_SRCS third-party/src/impl/list_ports/list_ports_linux.cc)
else()
    # If windows
    list(APPEND serial_SRCS third-party/src/impl/win.cc)
    list(APPEND serial_SRCS third-party/src/impl/list_ports/list_ports_win.cc)
endif()

## Add serial library
add_library(serial ${serial_SRCS})
if(APPLE)
    target_link_libraries(serial ${FOUNDATION_LIBRARY} ${IOKIT_LIBRARY})
elseif(UNIX)
    target_link_libraries(serial rt pthread)
else()
    target_link_libraries(serial setupapi)
endif()

# Define the executable
add_executable(IWR6843TestApp src/main.cpp)
add_executable(IWRApp src/IWRAPP.cpp)

# Link the executable with the shared library
target_link_libraries(IWR6843TestApp IWR6843Interface)
target_link_libraries(IWRApp IWR6843Interface serial)
