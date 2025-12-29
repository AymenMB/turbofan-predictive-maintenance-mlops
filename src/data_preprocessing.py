"""
Data preprocessing module for NASA CMAPSS FD001 dataset.

This module handles loading, cleaning, and feature engineering for the turbofan
engine degradation dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_and_process_data(data_path: str = "data/raw/train_FD001.txt") -> pd.DataFrame:
    """
    Load and preprocess NASA CMAPSS FD001 training data.
    
    Steps:
    1. Load space-separated data without headers
    2. Calculate Remaining Useful Life (RUL) for each time step
    3. Drop sensors with constant values (zero variance)
    
    Args:
        data_path: Path to the raw training data file
        
    Returns:
        Cleaned DataFrame with RUL target column and useful features
    """
    # Define column names for the dataset
    # 1 unit ID + 1 time + 3 operational settings + 21 sensor measurements
    column_names = ['unit_nr', 'time_cycles'] + \
                   [f'setting_{i}' for i in range(1, 4)] + \
                   [f's_{i}' for i in range(1, 22)]
    
    # Load the data
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=column_names)
    
    # Calculate RUL (Remaining Useful Life)
    # For each unit, find the maximum time_cycles (failure point)
    # Then calculate RUL = max_time - current_time for each row
    df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    
    # Drop sensors with constant values (zero standard deviation in FD001)
    # These sensors provide no information for prediction
    constant_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    df = df.drop(columns=constant_sensors)
    
    print(f"✓ Loaded {len(df)} rows from {data_path}")
    print(f"✓ Dataset contains {df['unit_nr'].nunique()} unique engines")
    print(f"✓ Dropped {len(constant_sensors)} constant sensors")
    print(f"✓ Final shape: {df.shape}")
    
    return df


if __name__ == "__main__":
    # Test the preprocessing function
    print("=" * 60)
    print("Testing NASA CMAPSS Data Preprocessing")
    print("=" * 60)
    
    # Load and process the data
    df = load_and_process_data()
    
    print("\n" + "=" * 60)
    print("First 5 rows:")
    print("=" * 60)
    print(df.head())
    
    print("\n" + "=" * 60)
    print("DataFrame Info:")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nRUL Statistics:")
    print(df['RUL'].describe())
