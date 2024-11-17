# Filename: correlation_database.py

import sqlite3
import logging
from typing import Optional, List, Tuple

class CorrelationDatabase:
    def __init__(self, db_path: str):
        """
        Initializes the CorrelationDatabase instance.

        Args:
            db_path (str): Path to the correlation SQLite database file.
        """
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path)
        self.create_table()

    def create_table(self) -> None:
        """
        Creates the 'correlations' table if it does not exist.
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS correlations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator TEXT NOT NULL,
            lag INTEGER NOT NULL,
            correlation REAL NOT NULL,
            date TEXT NOT NULL,
            UNIQUE(indicator, lag)
        );
        """
        create_index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_indicator ON correlations (indicator);",
            "CREATE INDEX IF NOT EXISTS idx_lag ON correlations (lag);"
        ]
        cursor = self.connection.cursor()
        cursor.execute(create_table_query)
        for query in create_index_queries:
            cursor.execute(query)
        self.connection.commit()

    def insert_correlation(self, indicator: str, lag: int, correlation: float, date: str) -> None:
        """
        Inserts a correlation record into the database.

        Args:
            indicator (str): Name of the indicator.
            lag (int): Lag period.
            correlation (float): Correlation value.
            date (str): Timestamp of calculation.
        """
        insert_query = """
        INSERT OR REPLACE INTO correlations (indicator, lag, correlation, date)
        VALUES (?, ?, ?, ?);
        """
        cursor = self.connection.cursor()
        try:
            cursor.execute(insert_query, (indicator, lag, correlation, date))
            self.connection.commit()
            logging.debug(f"Inserted correlation: {indicator}, Lag: {lag}, Correlation: {correlation}")
        except sqlite3.Error as e:
            logging.error(f"Failed to insert correlation: {e}")

    def get_correlation(self, indicator: str, lag: int) -> Optional[float]:
        """
        Retrieves a correlation value for a specific indicator and lag.

        Args:
            indicator (str): Name of the indicator.
            lag (int): Lag period.

        Returns:
            Optional[float]: Correlation value if found, else None.
        """
        select_query = """
        SELECT correlation FROM correlations
        WHERE indicator = ? AND lag = ?;
        """
        cursor = self.connection.cursor()
        cursor.execute(select_query, (indicator, lag))
        result = cursor.fetchone()
        return result[0] if result else None

    def get_all_correlations(self) -> List[Tuple[str, int, float, str]]:
        """
        Retrieves all correlation records from the database.

        Returns:
            List[Tuple[str, int, float, str]]: List of correlation records.
        """
        select_query = "SELECT indicator, lag, correlation, date FROM correlations;"
        cursor = self.connection.cursor()
        cursor.execute(select_query)
        return cursor.fetchall()

    def close(self) -> None:
        """
        Closes the database connection.
        """
        self.connection.close()
