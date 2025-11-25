"""
This script runs the IWRApp.exe application and annotates BLE acquisition in the database.
It uses subprocess to execute mmWave radar data acquisition and SQLAlchemy ORM to annotate and fetch BLE data.
It reuses a modular ORM reflection structure with automap.
At the end of the experiment, it stops the mmWave app, performs a query to retrieve BLE data, and saves it to CSV.

Usage:
py run_experiment.py --test_name "Test01" --test_description "Walking from point A to B"
"""

import subprocess
import os
import argparse
from time import sleep
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import pandas as pd

# === Reuse helpers from the modular ORM system ===
from sigivest.ble_query_helper import (
    load_database_url,
    reflect_all_schemas,
    automap_classes,
    fetch_beacon_data,
    start_test,
    end_test
)

def run_experiment(test_name, test_description):
    mmwave_data_output = f"{test_name}_mmwave_data.csv"
    ble_data_oputput = f"{test_name}_ble_data.csv"

    print(f"Test: {test_name} - {test_description}")
    input("Press Enter to start...")
    sleep(8) # Time for user to prepare for the experiment.
    print("Starting mmWave application...")
    mmwave_app_path = os.path.join(os.path.dirname(__file__), "build", "src", "SensorFusionApp.exe")
    process = subprocess.Popen([mmwave_app_path, '-o', mmwave_data_output])
    sleep(1)

    engine = create_engine(load_database_url())
    metadata = reflect_all_schemas(engine)
    Base = automap_classes(metadata)
    session = Session(bind=engine)
    session.execute(text("SET statement_timeout = 300000"))

    try:
        TEST_TIME = 350  # seconds of experiment duration
        print("Annotating BLE acquisition in the database...")
        start_test(session, test_name, test_description)
        sleep(TEST_TIME)
        end_test(session, test_name)
        process.terminate()
        print("mmWave application terminated.")

        print("Querying BLE data and saving to CSV...")
        results = fetch_beacon_data(session, Base, test_name)
        ble_data_df = pd.DataFrame(results)
        ble_data_df.to_csv(ble_data_oputput, index=False)

        print("Experiment completed successfully.")
        results_dir = os.path.join(os.path.dirname(__file__), "Results")
        os.makedirs(results_dir, exist_ok=True)
        output_dir = os.path.join(results_dir, test_name)
        os.makedirs(output_dir, exist_ok=True)
        os.rename(mmwave_data_output, os.path.join(output_dir, mmwave_data_output))
        os.rename(ble_data_oputput, os.path.join(output_dir, ble_data_oputput))

    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
        if process.poll() is None:
            process.terminate()
        raise e

    finally:
        session.close()
        if process.poll() is None:
            process.terminate()
            print("mmWave application terminated.")
        print("Stopped mmWave application.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run mmWave + BLE acquisition experiment")
    parser.add_argument("--test_name", required=True, help="Unique name for the test")
    parser.add_argument("--test_description", required=True, help="Description of the test scenario")
    args = parser.parse_args()

    run_experiment(args.test_name, args.test_description)
    print("Experiment script finished.")
