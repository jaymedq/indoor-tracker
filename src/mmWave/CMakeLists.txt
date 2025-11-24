# Define the source files for the mmWave component
set(MMWAVE_SRCS
    parser.cpp
    IWRAPP.cpp
    frames/AreaScannerFrame.cpp
)

# Define the mmWave static library
add_library(mmWaveLib STATIC ${MMWAVE_SRCS})

# Set include directories for this library. 
# We use PRIVATE because only its source files need to know about the project-wide headers.
# CMAKE_SOURCE_DIR points to project-root.
target_include_directories(mmWaveLib 
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/mmWave
        ${CMAKE_SOURCE_DIR}/include/mmWave/frames
        ${CMAKE_SOURCE_DIR}/third-party/include/serial
)

# Link the serial library, as the mmWave interface uses it internally
# Use PRIVATE because the mmWaveLib exposes data, not the serial port itself.
target_link_libraries(mmWaveLib PRIVATE serial)