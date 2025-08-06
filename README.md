# Indoor Tracker System - Sigivest Project

This project is part of the Sigivest research initiative at the Laboratory of Communication Systems (LabSC) at UTFPR. It implements an indoor tracking system that fuses data from a millimeter-wave (mmWave) radar sensor and Bluetooth Low Energy (BLE) beacons to accurately track the position of people within a room.

## Hardware

The system is designed to work with the following hardware:
- **mmWave Sensor:** TEXAS INSTRUMENTS IWR6843ISK
- **BLE Beacon:** SILABS BRD4191A BG22

## Project Structure

The repository is organized as follows:

```
├── .vscode/                # VSCode workspace settings
├── 3d/                     # 3D modeling files for sensor mounts
├── build/                  # Build output for the C++ application
├── cfg/                    # Configuration files for the mmWave sensor
├── parser_scripts/         # Python scripts for parsing sensor data
├── Results/                # Output directory for experiment data
├── sigivest/               # Python module for database interaction (BLE)
├── src/                    # C++ source code for the mmWave data acquisition app (IWRApp)
├── third-party/            # Third-party libraries (e.g., serial communication)
├── tools/                  # Python scripts for data processing, fusion, and analysis
├── .env                    # Environment variables (e.g., database URL)
├── .gitignore              # Git ignore file
├── CMakeLists.txt          # Main CMake build script for the C++ application
├── requirements.txt        # Python dependencies
├── run_experiment.py       # Main script to orchestrate data collection experiments
└── README.md               # This file
```

## System Workflow & Usage

The process from data acquisition to analysis is divided into several steps.

### 1. Prerequisites

**A. Build the C++ Application:**

The `IWRApp.exe` is required to collect data from the mmWave sensor. Build it using CMake and a compatible C++ compiler (e.g., MinGW, MSVC).

```bash
# Create a build directory
mkdir build
cd build

# Configure and build the project
cmake ..
cmake --build .
```
This will generate `IWRApp.exe` inside the `build` directory.

**B. Install Python Dependencies:**

The data processing and analysis scripts are written in Python. Install the required libraries using pip:
```bash
pip install -r requirements.txt
```

**C. Configure Database:**

The system uses a database to store and retrieve BLE data. Create a `.env` file in the project root and define the database connection URL:
```
DATABASE_URL="postgresql://user:password@host:port/database"
```

### 2. Data Acquisition

The `run_experiment.py` script automates the data collection process. It starts the `IWRApp.exe` to capture mmWave data and simultaneously annotates the start and end times of the experiment in the BLE database.

To run an experiment, use the following command:
```bash
py run_experiment.py --test_name "MyTest01" --test_description "User walking in a straight line"
```
- `--test_name`: A unique identifier for the experiment.
- `--test_description`: A brief description of the test scenario.

The script will create a directory inside `Results/` named after your `test_name`, containing the raw mmWave data and the queried BLE data.

### 3. Data Processing and Fusion

After acquiring the data, you need to process it. The following scripts must be run in order:

**A. Pre-process the Dataset:**
This script (if applicable) cleans and prepares the raw data from the experiment for the next steps.
```bash
py tools/pre_process_dataset.py
```

**B. Fuse Sensor Data:**
This script fuses the mmWave and BLE data using a Kalman filter to produce a more accurate, unified trajectory.
```bash
py tools/fuse_sensor_data.py
```

**C. Analyze Kalman Filter Divergence:**
This script analyzes the filter's performance and potential divergence issues.
```bash
py tools/kf_divergence.py
```

### 4. Visualization and Results Analysis

Finally, use the plotting scripts to visualize and evaluate the results.

**A. Plot Animated Trajectory:**
This script creates an MP4 video animating the real, measured, and fused trajectories.
```bash
py tools/plot_dataset_animation.py
```

**B. Plot Error Metrics:**
This script generates plots showing the Mean Squared Error (MSE) and Root Mean Squared Error (RMSE) of the tracking system relative to the ground truth.
```bash
py tools/plot_mse_rmse_by_distance.py
```

## Key Scripts

- `run_experiment.py`: Orchestrates the entire data acquisition process.
- `src/main.cpp`: The core of the `IWRApp` that connects to and parses data from the mmWave sensor.
- `tools/fuse_sensor_data.py`: Implements the track-to-track fusion algorithm using a 2D Kalman filter for both BLE and mmWave data streams.
- `sigivest/ble_query_helper.py`: Contains helper functions to interact with the BLE database using SQLAlchemy.

## License

This project is licensed under the terms of the LICENSE file.

## Acknowledgments

This work is supported by the **Sigivest project** at the **Laboratory of Communication Systems (LabSC)**, Federal University of Technology - Parana (UTFPR).
