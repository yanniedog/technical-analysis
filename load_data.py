# load_data.py

import os
import sys
import sqlite3
import logging
import pandas as pd
from dateutil import parser
import numpy as np

# Import the database path from config.py
from config import DB_PATH

def create_database(db_path: str) -> None:
    """
    Creates the SQLite database with the required schema if it doesn't exist.

    Args:
        db_path (str): The full path to the SQLite database file.
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Create the 'klines' table with the required schema
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS klines (
                open_time TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                close_time TEXT,
                quote_asset_volume REAL,
                number_of_trades INTEGER,
                taker_buy_base_asset_volume REAL,
                taker_buy_quote_asset_volume REAL,
                correlation_computed BOOLEAN DEFAULT FALSE
            )
        ''')
        conn.commit()
        conn.close()
        logging.info(f"Database created with schema at: {db_path}")
    except sqlite3.Error as e:
        logging.error(f"Failed to create database '{db_path}'. Error: {e}")
        sys.exit(1)

def load_data():
    """
    Loads and preprocesses data from the SQLite database.

    Returns:
        tuple: A tuple containing the processed DataFrame, a boolean indicating if the data is in reverse chronological order, and the database filename.
    """
    # Import logging configuration if needed
    logging.basicConfig(level=logging.INFO)

    # Use the database path from config.py
    db_path = DB_PATH

    # Check if the database file exists; if not, create it with the required schema
    if not os.path.exists(db_path):
        logging.warning(f"SQLite database '{db_path}' not found. Creating a new database.")
        create_database(db_path)

    # Connect to the database
    try:
        conn = sqlite3.connect(db_path)
        logging.info(f"Connected to SQLite database at: {db_path}")
    except sqlite3.Error as e:
        logging.error(f"Failed to connect to database '{db_path}'. Error: {e}")
        sys.exit(1)

    # Define the SQL query to retrieve data
    query = "SELECT * FROM klines ORDER BY open_time ASC"

    try:
        # Parse 'open_time' and 'close_time' columns as datetime
        data = pd.read_sql_query(query, conn, parse_dates=['open_time', 'close_time'])
        logging.info("Data loaded from SQLite database.")
    except pd.io.sql.DatabaseError as e:
        logging.error(f"Failed to execute query. Error: {e}")
        conn.close()
        sys.exit(1)

    # Close the database connection
    conn.close()

    # Ensure the data is not empty
    if data.empty:
        logging.warning("No data available in the database.")
        is_reverse_chronological = False  # Assign a default value
        # Return the empty DataFrame and indicate that the data is not reverse chronological
        return data, is_reverse_chronological, os.path.basename(db_path)
    else:
        # Determine if the data is sorted in reverse chronological order
        time_column = 'open_time'
        is_reverse_chronological = data[time_column].is_monotonic_decreasing

        # Sort the data in chronological order if it's in reverse
        if is_reverse_chronological:
            data = data.sort_values(time_column)
            data.reset_index(drop=True, inplace=True)
            logging.info("Data sorted in chronological order.")

        # Drop any missing data
        data.dropna(inplace=True)

        logging.info("Columns in the data: %s", list(data.columns))
        logging.info("First few rows of the data:\n%s", data.head())

        # Remove timezone information from 'open_time' and 'close_time'
        data['open_time'] = data['open_time'].dt.tz_localize(None)
        data['close_time'] = data['close_time'].dt.tz_localize(None)
        logging.info("Removed timezone information from 'open_time' and 'close_time'.")

        # Ensure 'open_time' is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(data['open_time']):
            data['open_time'] = pd.to_datetime(data['open_time'], errors='coerce')

        # Determine the time interval between rows
        try:
            data['TimeDiff'] = data['open_time'].diff().dt.total_seconds()
            if data['TimeDiff'].isna().all():
                logging.error("Failed to determine time interval: All computed time differences are NaN.")
                sys.exit(1)
            else:
                logging.info("First few time differences (in seconds):\n%s", data['TimeDiff'].head())
                # Fill the NaN value for the first row with zero or drop it
                data['TimeDiff'].fillna(0, inplace=True)
        except Exception as e:
            logging.error(f"Failed to compute time differences: {e}")
            sys.exit(1)

    # Return the processed DataFrame, the reverse chronological order flag, and the database filename
    return data, is_reverse_chronological, os.path.basename(db_path)

if __name__ == "__main__":
    df, is_rev, filename = load_data()
    # You can add further processing or function calls here as needed
