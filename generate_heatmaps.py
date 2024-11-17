# filename: generate_heatmaps.py

import os
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import pandas as pd
from joblib import Parallel, delayed

def generate_heatmaps(data, timestamp, time_interval, generate_heatmaps, cache, calculate_correlation, base_csv_filename):
    """
    Generates a single heatmap combining the correlation data from the entire range of indicators.

    Args:
        data (pd.DataFrame): DataFrame containing all data with indicators.
        timestamp (str): Current timestamp in YYYYMMDD-HHMMSS format for filename prefixing.
        time_interval (str): Time interval between rows (e.g., 'minute', 'hour', 'day', 'week').
        generate_heatmaps (bool): Boolean indicating if heatmaps should be generated.
        cache (dict): Cache dictionary to store computed correlations.
        calculate_correlation (function): Function to calculate correlation for a given indicator and lag.
        base_csv_filename (str): Base filename of the original CSV file.
    """
    if not generate_heatmaps:
        return

    heatmaps_dir = 'heatmaps'
    if not os.path.exists(heatmaps_dir):
        os.makedirs(heatmaps_dir)
    else:
        existing_files = os.listdir(heatmaps_dir)
        if existing_files:
            logging.info(f"Existing files found in '{heatmaps_dir}' directory.")
            delete_choice = input(f"Do you want to delete them? (y/n): ").lower()
            if delete_choice == 'y':
                for file in existing_files:
                    file_path = os.path.join(heatmaps_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                logging.info(f"Deleted existing heatmaps in '{heatmaps_dir}'.")
            else:
                logging.info(f"Retaining existing heatmaps in '{heatmaps_dir}'.")

    # Filter out original indicators, explicitly excluding 'Close'
    original_indicators = [
        col for col in data.columns 
        if pd.api.types.is_numeric_dtype(data[col]) 
           and col != 'Close'  # Exclude 'Close'
           and data[col].notna().any() 
           and data[col].var() > 1e-6
    ]

    # Debug statement to confirm exclusion
    print(f"Original indicators in generate_heatmaps.py: {original_indicators}")

    # Calculate correlations for each indicator, starting from lag=1
    max_lag = len(data) - 51  # Maximum lag to consider (50 less than total rows)
    correlations = {}
    for col in original_indicators:
        if col not in cache:
            # Start lags from 1 to avoid lag=0
            corr_list = Parallel(n_jobs=-1)(
                delayed(calculate_correlation)(data, col, lag, False) 
                for lag in range(1, max_lag + 1)
            )
            cache[col] = corr_list
        else:
            corr_list = cache[col]
        correlations[col] = corr_list

    # Create a DataFrame for the correlations
    corr_df = pd.DataFrame(correlations, index=range(1, max_lag + 1))  # Start from lag=1

    # Filter out rows and columns with all NaN values
    corr_df = corr_df.dropna(axis=1, how='all').dropna(axis=0, how='all')

    # Standardize each row (indicator) to have max value of 1.0 and min value of -1.0
    def standardize_row(row):
        if row.max() - row.min() == 0:
            return row * 0  # Avoid division by zero if all values are the same
        return (row - row.min()) / (row.max() - row.min()) * 2 - 1

    standardized_corr_df = corr_df.apply(standardize_row, axis=1)

    # Filter indicators by max correlation exceeding 0.25
    filtered_indicators = [
        col for col in standardized_corr_df.columns 
        if standardized_corr_df[col].max() > 0.25
    ]
    standardized_corr_df = standardized_corr_df[filtered_indicators]

    # Sort indicators based on the earliest lag time in which a 1.0 correlation occurs
    sorted_indicators_1 = sorted(
        filtered_indicators, 
        key=lambda col: next((i for i, x in enumerate(standardized_corr_df[col]) if x == 1.0), max_lag)
    )
    sorted_standardized_corr_df_1 = standardized_corr_df[sorted_indicators_1]

    # Plot the first heatmap
    plt.figure(figsize=(20, 15), dpi=300)
    sns.heatmap(
        sorted_standardized_corr_df_1.T, 
        annot=False, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=True, 
        yticklabels=True
    )
    plt.title('Standardized Correlation of Indicators with Close Price at Various Time Lags (Sorted by Earliest 1.0 Correlation)', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)  # Rotate x-axis labels by 90 degrees
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    heatmap_filename_1 = f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_1.png"
    heatmap_filepath_1 = os.path.join(heatmaps_dir, heatmap_filename_1)
    plt.savefig(heatmap_filepath_1, bbox_inches='tight')
    plt.close()

    # Sort indicators based on the highest correlation at lag time 1
    sorted_indicators_2 = sorted(
        filtered_indicators, 
        key=lambda col: standardized_corr_df[col].iloc[0], 
        reverse=True
    )
    sorted_standardized_corr_df_2 = standardized_corr_df[sorted_indicators_2]

    # Plot the second heatmap
    plt.figure(figsize=(20, 15), dpi=300)
    sns.heatmap(
        sorted_standardized_corr_df_2.T, 
        annot=False, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=True, 
        yticklabels=True
    )
    plt.title('Standardized Correlation of Indicators with Close Price at Various Time Lags (Sorted by Highest Correlation at Lag 1)', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)  # Rotate x-axis labels by 90 degrees
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    heatmap_filename_2 = f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_2.png"
    heatmap_filepath_2 = os.path.join(heatmaps_dir, heatmap_filename_2)
    plt.savefig(heatmap_filepath_2, bbox_inches='tight')
    plt.close()

    # Generate heatmap with raw values
    raw_corr_df = corr_df[filtered_indicators]
    sorted_indicators_3 = sorted(
        filtered_indicators, 
        key=lambda col: raw_corr_df[col].iloc[0], 
        reverse=True
    )
    sorted_raw_corr_df = raw_corr_df[sorted_indicators_3]

    # Plot the third heatmap
    plt.figure(figsize=(20, 15), dpi=300)
    sns.heatmap(
        sorted_raw_corr_df.T, 
        annot=False, 
        cmap='coolwarm', 
        cbar=True, 
        xticklabels=True, 
        yticklabels=True
    )
    plt.title('Raw Correlation of Indicators with Close Price at Various Time Lags (Sorted by Highest Correlation at Lag 1)', fontsize=14)
    plt.xlabel(f'Time Lag ({time_interval})', fontsize=12)
    plt.ylabel('Indicators', fontsize=12)
    plt.xticks(rotation=90, fontsize=6)  # Rotate x-axis labels by 90 degrees
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    heatmap_filename_3 = f"{timestamp}_{base_csv_filename}_combined_correlation_heatmap_3.png"
    heatmap_filepath_3 = os.path.join(heatmaps_dir, heatmap_filename_3)
    plt.savefig(heatmap_filepath_3, bbox_inches='tight')
    plt.close()

    logging.info(f"Generated combined correlation heatmaps in '{heatmaps_dir}'.")