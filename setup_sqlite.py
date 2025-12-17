
import sqlite3
import pandas as pd
import os

DB_NAME = 'water_demand.db'

def create_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_NAME)
        print(f"Connected to SQLite database: {DB_NAME}")
    except sqlite3.Error as e:
        print(f"Error connecting to SQLite: {e}")
    return conn

def setup_database():
    conn = create_connection()
    if conn is None:
        return

    cursor = conn.cursor()
    
    try:
        # Create water_audit_dataset table
        cursor.execute("DROP TABLE IF EXISTS water_audit_dataset")
        create_table_query = """
        CREATE TABLE water_audit_dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            did TEXT,
            month_year TEXT,
            rainfall REAL,
            ground_water REAL,
            soil_moisture REAL,
            major REAL,
            medium REAL,
            mi_tanks REAL,
            evapotranspiration REAL,
            outflow REAL,
            consumption REAL,
            target REAL
        )
        """
        cursor.execute(create_table_query)
        print("Table 'water_audit_dataset' created.")
        
        # Insert dummy data for all districts and reservoirs
        # Districts: d1 (Ananthapuram), d2 (Chittor), d3 (East Godavari), d5 (Krishna), d6 (Kurnool)
        # Reservoirs: r1 (Chemmbarambakkam), r2 (Cholavaram)
        
        districts = ['d1', 'd2', 'd3', 'd5', 'd6']
        # Load CSV data
        csv_path = os.path.join(os.getcwd(), 'templates', 'Water Audit final 100 years data.csv')
        
        val = []
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Replicate CSV data for EACH district for demo purposes
            for d_id in districts:
                for index, row in df.iterrows():
                    val.append((
                        d_id, 
                        row['month-year'],
                        row['Rainfall'],
                        row['Ground Water'],
                        row['Soil Moisture'],
                        row['Major'],
                        row['Medium'],
                        row['MI Tanks'],
                        row['Evapotranspiration'],
                        row['Surface and SubSurface Outflow'],
                        row['Consumption'],
                        row['Target']
                    ))
        else:
             # Fallback dummy data if CSV missing
             for d_id in districts:
                 for i in range(12):
                     val.append((d_id, f"Jan-{2014+i}", 10.5, 5.0, 2.0, 1.0, 0.5, 0.2, 5.0, 1.0, 8.0, 10.0))

        sql = """
        INSERT INTO water_audit_dataset 
        (did, month_year, rainfall, ground_water, soil_moisture, major, medium, mi_tanks, evapotranspiration, outflow, consumption, target) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(sql, val)
        conn.commit()
        print(f"Inserted {len(val)} rows into 'water_audit_dataset'.")

        # Create chemmbarambakkam table
        cursor.execute("DROP TABLE IF EXISTS chemmbarambakkam")
        create_res_table_query = """
        CREATE TABLE chemmbarambakkam (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rid TEXT,
            date TEXT,
            level REAL,
            storage REAL,
            inflow REAL,
            outflow REAL
        )
        """
        cursor.execute(create_res_table_query)
        print("Table 'chemmbarambakkam' created.")
        
        # Insert dummy data for Reservoirs
        res_data = []
        reservoirs = ['r1', 'r2', 'rid'] # 'rid' included because it's the default value in some dropdowns potentially
        
        for r_id in reservoirs:
            for i in range(1, 31):
                res_data.append((r_id, f'2024-01-{i:02d}', 20.0 + (i%5), 3000.0 + (i*10), 100.0 + (i*5), 50.0 + (i*2)))
        
        sql_res = "INSERT INTO chemmbarambakkam (rid, date, level, storage, inflow, outflow) VALUES (?, ?, ?, ?, ?, ?)"
        cursor.executemany(sql_res, res_data)
        conn.commit()
        print("Inserted dummy data into 'chemmbarambakkam'.")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()
            print("SQLite connection closed.")

if __name__ == "__main__":
    setup_database()
