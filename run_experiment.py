"""
This script runs the IWRApp.exe application and annotates BLE acquisition in the database.
It uses subprocess to execute mmWave radar data acquisition and SQLAlchemy ORM to annotate and fetch BLE data.
It reuses a modular ORM reflection structure with automap.
At the end of the experiment, it stops the mmWave app, performs a query to retrieve BLE data, and saves it to CSV.
"""

import subprocess
import os
from time import sleep
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# === Reuse helpers from the modular ORM system ===
from sigivest.ble_query_helper import (
    load_database_url,
    reflect_all_schemas,
    automap_classes,
    fetch_beacon_data,
    start_test,
    end_test
)

def run_experiment():
    test_name = input("Type a name for the test:\n")
    test_description = input("Type a description of the test:\n")
    mmwave_data_output = f"{test_name}_mmwave_data.csv"
    ble_data_oputput = f"{test_name}_ble_data.csv"

    print(f"Test: {test_name} - {test_description}")
    input("Press Enter to start...")
    sleep(6) # Time for user to prepare for the experiment.
    print("Starting mmWave application...")
    mmwave_app_path = os.path.join(os.path.dirname(__file__),"build","IWRApp.exe")
    #Call IRWApp with output file name as parameter
    process = subprocess.Popen([mmwave_app_path, '-o', mmwave_data_output],)
    # Wait script startu
    sleep(1)

    engine = create_engine(load_database_url())
    metadata = reflect_all_schemas(engine)
    Base = automap_classes(metadata)
    session = Session(bind=engine)
    session.execute(text("SET statement_timeout = 300000"))

    try:
        TEST_TIME = 120  # seconds of experiment duration
        print("Annotating BLE acquisition in the database...")
        start_test(session, test_name, test_description)
        sleep(TEST_TIME)
        # input("Press Enter to end the test...")
        end_test(session, test_name)
        process.terminate()
        print("mmWave application terminated.")

        print("Querying BLE data and saving to CSV...")
        # results = fetch_beacon_data(session, Base, test_name)
        # ble_data_df = pd.DataFrame(results)
        # ble_data_df.to_csv(ble_data_oputput, index=False)

        print("Experiment completed successfully.")
        # move output files to a subdirectory named after the test inside Results folder
        results_dir = os.path.join(os.path.dirname(__file__), "Results")
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        output_dir = os.path.join(results_dir,test_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        os.rename(mmwave_data_output, os.path.join(output_dir, mmwave_data_output))
        # os.rename(ble_data_oputput, os.path.join(output_dir, ble_data_oputput))

    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()
        raise e

    finally:
        session.close()
        if(process.poll() is None):
            process.terminate()
            print("mmWave application terminated.")
        print("Stopped mmWave application.")

if __name__ == "__main__":
    run_experiment()
    print("Experiment script finished.")
