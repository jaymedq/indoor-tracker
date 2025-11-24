# project-root/src/CMakeLists.txt

# 1. Include all subdirectories to define the libraries (e.g., mmWaveLib, IMULib, etc.)
add_subdirectory(mmWave) 
add_subdirectory(imu) 
add_subdirectory(ble)
add_subdirectory(sensor_fusion)


# 2. Define the main executable
add_executable(SensorFusionApp main.cpp)


# 3. Set project-wide includes for the application
# This ensures main.cpp can find headers like "Fusion/FusionAlgorithm.h"
target_include_directories(SensorFusionApp 
    PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/include/mmWave
        ${CMAKE_SOURCE_DIR}/include/mmWave/frames
        ${CMAKE_SOURCE_DIR}/third-party/include
)


# 4. Link all defined libraries into the final executable
target_link_libraries(SensorFusionApp 
    PRIVATE 
        mmWaveLib
)