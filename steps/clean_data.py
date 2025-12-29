"""
Data cleaning step for ZenML pipeline.

Calculates RUL and removes constant sensors.
"""

import pandas as pd
from zenml import step


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess NASA CMAPSS data.
    
    Steps:
    1. Calculate Remaining Useful Life (RUL) for each time step
    2. Drop sensors with constant values (zero variance)
    
    Args:
        df: Raw DataFrame from ingestion step
        
    Returns:
        Cleaned DataFrame with RUL target column and useful features
    """
    # Calculate RUL (Remaining Useful Life)
    # For each unit, find the maximum time_cycles (failure point)
    # Then calculate RUL = max_time - current_time for each row
    df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    
    # Drop sensors with constant values (zero standard deviation in FD001)
    # These sensors provide no information for prediction
    constant_sensors = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
    df = df.drop(columns=constant_sensors)
    
    print(f"✓ Calculated RUL for all samples")
    print(f"✓ Dropped {len(constant_sensors)} constant sensors")
    print(f"✓ Final shape: {df.shape}")
    
    return df
