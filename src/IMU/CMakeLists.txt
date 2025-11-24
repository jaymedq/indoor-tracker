# src/imu/CMakeLists.txt

# Define the IMU interface library
add_library(IMULib INTERFACE)

# Set include directories for this library
target_include_directories(IMULib 
    INTERFACE 
        ${CMAKE_CURRENT_SOURCE_DIR}
)
