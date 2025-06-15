"""
This script runs the IWRApp.exe application and annotates BLE acquisition in the database.
It uses subprocess to execute mmWave radar data acquisition and SQLAlchemy ORM to annotate and fetch BLE data.
It reuses a modular ORM reflection structure with automap.
At the end of the experiment, it stops the mmWave app, performs a query to retrieve BLE data, and saves it to CSV.
"""

import subprocess
import os
import time
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

# === Reuse helpers from the modular ORM system ===
from sigivest.ble_query_helper import (
    load_database_url,
    reflect_all_schemas,
    automap_classes,
    fetch_beacon_data
)

def run_experiment():
    mmwave_app_path = "C:\\Users\\sigivest\\Desktop\\IWRApp.exe"
    test_name = "TesteJayme1"

    print("Starting mmWave application...")
    # process = subprocess.Popen([mmwave_app_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time.sleep(3)  # Ensure app starts

    try:
        engine = create_engine(load_database_url())
        metadata = reflect_all_schemas(engine)
        Base = automap_classes(metadata)
        session = Session(bind=engine)
        session.execute(text("SET statement_timeout = 300000"))

        # Optional: Insert BLE annotation (placeholder)
        print("Annotating BLE acquisition in the database...")
        # Example: session.add(SomeNewEntry(...))
        # session.commit()

        # === Query and Export BLE data ===
        print("Querying BLE data and saving to CSV...")
        results = fetch_beacon_data(session, Base, test_name)
        ble_data_df = pd.DataFrame(results)
        ble_data_df.to_csv("ble_data.csv", index=False)

        print("Experiment completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")
        session.rollback()

    finally:
        session.close()
        # process.terminate()
        print("Stopped mmWave application.")

if __name__ == "__main__":
    run_experiment()
    print("Experiment script finished.")
