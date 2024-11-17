# Filename: correlation_utils.py

import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
import sqlite3

def calculate_correlation(
    data: pd.DataFrame,
    col: str,
    lag: int,
    is_reverse_chronological: bool
) -> float:
    """
    Calculates the correlation between a shifted indicator and the 'Close' price.

    Args:
        data (pd.DataFrame): The dataset containing indicators and 'Close' price.
        col (str): The indicator column name.
        lag (int): The lag period.
        is_reverse_chronological (bool): Indicates if data is in reverse chronological order.

    Returns:
        float: The correlation coefficient or NaN if invalid.
    """
    logging.debug(f"Starting correlation for '{col}' at lag {lag}.")
    if col == 'Close':
        logging.warning(f"Skipping correlation calculation for '{col}' at lag {lag} as it is the target variable.")
        return np.nan
    try:
        shift_value = lag if is_reverse_chronological else -lag
        shifted_col = data[col].shift(shift_value)
        valid_data = pd.concat([shifted_col, data['Close']], axis=1).dropna()
        if not valid_data.empty:
            corr = valid_data[col].corr(valid_data['Close'])
            logging.debug(f"Correlation for '{col}' at lag {lag}: {corr}")
            return corr
        logging.warning(f"No valid data for '{col}' at lag {lag}'. Returning NaN.")
        return np.nan
    except Exception as e:
        logging.error(f"Error calculating correlation for {col} at lag {lag}: {e}")
        return np.nan

def load_cache(cache_filename: str) -> Dict[str, Any]:
    """
    Loads the cache from a JSON file if it exists.

    Args:
        cache_filename (str): Path to the cache JSON file.

    Returns:
        Dict[str, Any]: The loaded cache or an empty dictionary if loading fails.
    """
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, "r") as f:
                cache = json.load(f)
            logging.info(f"Loaded cache from '{cache_filename}'.")
            return convert_cache_for_json(cache)
        except Exception as e:
            logging.error(f"Failed to load cache file '{cache_filename}': {e}")
    else:
        logging.info(f"No cache file found at '{cache_filename}'. A new cache will be created.")
    return {}

def save_cache(cache: Dict[str, Any], cache_filename: str) -> None:
    """
    Saves the cache dictionary to a JSON file atomically.

    Args:
        cache (Dict[str, Any]): The cache dictionary.
        cache_filename (str): Path to the cache JSON file.
    """
    temp_filename = f"{cache_filename}.tmp"
    try:
        with open(temp_filename, "w") as f:
            json.dump(cache, f, indent=4)
        os.replace(temp_filename, cache_filename)  # Atomic move
        logging.info(f"Cache saved to '{os.path.abspath(cache_filename)}'.")
    except Exception as e:
        logging.error(f"Failed to save cache to '{cache_filename}': {e}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

def convert_cache_for_json(cache: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert cache data to types compatible with JSON serialization.

    Args:
        cache (Dict[str, Any]): The cache dictionary.

    Returns:
        Dict[str, Any]: The converted cache dictionary.
    """
    if 'dates' in cache:
        cache['dates'] = [str(date) for date in cache['dates']]

    if 'correlations' in cache:
        for key in cache['correlations']:
            cache['correlations'][key] = [
                float(corr) if isinstance(corr, np.generic) else corr for corr in cache['correlations'][key]
            ]

    return cache

def is_valid_indicator(series: pd.Series) -> bool:
    """
    Checks if the indicator series is valid for correlation calculation.

    Args:
        series (pd.Series): The indicator data series.

    Returns:
        bool: True if valid, False otherwise.
    """
    # Check for sufficient non-NaN values
    if series.isna().all():
        logging.warning("Indicator series contains only NaN values.")
        return False
    # Check for variability (non-constant)
    if series.nunique() <= 1:
        logging.warning("Indicator series is constant.")
        return False
    return True

def load_or_calculate_correlations(
    data: pd.DataFrame,
    original_indicators: List[str],
    max_lag: int,
    is_reverse_chronological: bool,
    cache_filename: str
) -> Dict[str, List[float]]:
    """
    Loads cached correlations or calculates them if cache is invalid or absent.
    Progressively saves the cache after calculating correlations for each indicator.

    Args:
        data (pd.DataFrame): The dataset.
        original_indicators (List[str]): List of indicators to calculate correlations for.
        max_lag (int): Maximum lag period.
        is_reverse_chronological (bool): Data ordering flag.
        cache_filename (str): Path to cache file.

    Returns:
        Dict[str, List[float]]: Correlation data.
    """
    cache = load_cache(cache_filename)
    data_dates = data['Date'].tolist()
    cached_dates = cache.get('dates', [])

    if cached_dates == [str(date) for date in data_dates]:
        logging.info("Cache is valid. Using cached correlation data.")
        correlations = cache.get('correlations', {})
    else:
        if cached_dates:
            logging.info("Cache dates do not match current data. Recalculating correlations.")
        else:
            logging.info("No date information in cache. Recalculating correlations.")

        # Initialize correlations with existing cache to allow partial saving
        correlations = cache.get('correlations', {})
        # Update cache dates
        cache['dates'] = [str(date) for date in data_dates]

        for col in original_indicators:
            if col in correlations:
                logging.info(f"Indicator '{col}' already in cache. Skipping calculation.")
                continue  # Skip already cached indicators

            # Validate the indicator before processing
            if not is_valid_indicator(data[col]):
                logging.warning(f"Indicator '{col}' is invalid. Skipping correlation calculations.")
                correlations[col] = [np.nan] * max_lag
                # Update cache with the invalid indicator
                cache['correlations'] = correlations
                save_cache(cache, cache_filename)
                continue

            logging.info(f"Calculating correlations for indicator '{col}'.")

            try:
                # Calculate correlations in parallel
                corr_list = Parallel(n_jobs=4)(
                    delayed(calculate_correlation)(data, col, lag, is_reverse_chronological)
                    for lag in range(1, max_lag + 1)
                )
                # Convert correlations to float and handle NaNs
                correlations[col] = [float(corr) if isinstance(corr, (float, np.float64)) else np.nan for corr in corr_list]
                logging.info(f"Calculated correlations for '{col}'.")
            except Exception as e:
                logging.error(f"Failed to calculate correlations for '{col}': {e}")
                correlations[col] = [np.nan] * max_lag  # Assign NaNs for all lags in case of failure

            # Update cache with the newly calculated correlations
            cache['correlations'] = correlations
            # Save cache after processing each indicator
            save_cache(cache, cache_filename)

    # Ensure original_indicators are in correlations
    original_indicators = [col for col in original_indicators if col in correlations]
    logging.info(f"Updated original indicators after checking correlations: {original_indicators}")

    if not original_indicators:
        logging.error("No indicators with valid correlations found.")
        raise ValueError("No indicators with valid correlations found.")

    return correlations

def fetch_uncomputed_records():
    """
    Fetches records from the SQLite database where correlation_computed is FALSE.

    Returns:
        pd.DataFrame: DataFrame containing uncomputed records.
    """
    db_filename = 'klines.db'
    conn = sqlite3.connect(db_filename)
    query = "SELECT * FROM klines WHERE correlation_computed = FALSE ORDER BY open_time ASC"
    data = pd.read_sql_query(query, conn)
    conn.close()
    return data

def update_correlation_computed(records):
    """
    Updates the correlation_computed flag to TRUE for the provided records.

    Args:
        records (pd.DataFrame): DataFrame containing records to update.
    """
    db_filename = 'klines.db'
    conn = sqlite3.connect(db_filename)
    cursor = conn.cursor()
    for open_time in records['open_time']:
        cursor.execute("UPDATE klines SET correlation_computed = TRUE WHERE open_time = ?", (open_time,))
    conn.commit()
    conn.close()