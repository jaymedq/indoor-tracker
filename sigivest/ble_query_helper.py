import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine, MetaData, inspect, func, case, text, cast
)
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import UUID

VERBOSE = False  # Set to False to suppress schema/class output

def load_database_url():
    load_dotenv()
    return f"postgresql://{os.getenv('PGUSER')}:{os.getenv('PGPASSWORD')}@" \
           f"{os.getenv('PGHOST')}:{os.getenv('PGPORT')}/{os.getenv('PGDATABASE')}"

def reflect_all_schemas(engine):
    inspector = inspect(engine)
    schemas = [s for s in inspector.get_schema_names() if s not in ("information_schema", "pg_catalog")]
    metadata = MetaData()
    for schema in schemas:
        metadata.reflect(bind=engine, schema=schema)
    return metadata

def automap_classes(metadata):
    Base = automap_base(metadata=metadata)
    Base.prepare()
    return Base

def fetch_beacon_data(session, Base, test_name):
    B = Base.classes.beacons
    A = Base.classes.arrivals
    AOA = Base.classes.angle_of_arrivals
    CTE = Base.classes.cte_iq_samples
    PPEC = Base.classes.ppe_coordinates
    TA = Base.classes.test_annotations

    arr_ids = [
        '9055c740-ff71-11ef-b668-0242ac120009',
        '9055ab34-ff71-11ef-b668-0242ac120009',
        '9055c7cc-ff71-11ef-b668-0242ac120009',
        '9055c790-ff71-11ef-b668-0242ac120009',
    ]

    def case_max(col_expr, uuid_str, label):
        return func.max(case((A.idarray_coordinate == uuid_str, col_expr))).label(label)

    time_bounds = (
        session.query(TA.start_time, TA.end_time)
        .filter(TA.test_name == test_name)
        .subquery()
    )

    query = session.query(
        B.idbeacon,

        # RSSI
        *[case_max(A.rssi_dbm, arr_ids[i], f"rssi_{4 - i}") for i in range(4)],

        # AOA Alpha (azimuth)
        *[case_max(text("(arrival_angle_rad).alpha"), arr_ids[i], f"az_{4 - i}") for i in range(4)],

        # AOA Gamma (elevation)
        *[case_max(text("(arrival_angle_rad).gamma"), arr_ids[i], f"el_{4 - i}") for i in range(4)],

        # IQ Arrays
        *[case_max(CTE.i_array, arr_ids[i], f"i_{4 - i}") for i in range(4)],
        *[case_max(CTE.q_array, arr_ids[i], f"q_{4 - i}") for i in range(4)],

        # PPE coordinates
        func.max(text("(coordinate_m).x")).label("x"),
        func.max(text("(coordinate_m).y")).label("y"),
        func.max(text("(coordinate_m).z")).label("z"),

        B.create_time,
        B.bluetooth_channel,
        B.idppe
    ).outerjoin(A, B.idbeacon == A.idbeacon
    ).outerjoin(AOA, AOA.idarrival == A.idarrival
    ).outerjoin(CTE, CTE.idarrival == A.idarrival
    ).outerjoin(PPEC, cast(PPEC.metadata['beacon_uuid'].astext, UUID) == B.idbeacon
    ).filter(
        B.create_time >= time_bounds.c.start_time,
        B.create_time <= time_bounds.c.end_time,
        A.idarray_coordinate.in_(arr_ids)
    ).group_by(B.idbeacon).order_by(B.create_time)

    return query.all()

def main():
    engine = create_engine(load_database_url())
    metadata = reflect_all_schemas(engine)
    Base = automap_classes(metadata)

    session = Session(bind=engine)
    session.execute(text("SET statement_timeout = 300000"))

    results = fetch_beacon_data(session, Base, "TesteJayme1")
    df = pd.DataFrame(results)
    df.to_csv("ble_data_query_result.csv", index=False)
    print("CSV export complete: ble_data_query_result.csv")

if __name__ == "__main__":
    main()
