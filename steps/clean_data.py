"""
Clean data step for ZenML pipeline.

This step handles data cleaning and feature engineering:
- Calculate RUL (Remaining Useful Life)
- Clip RUL at 125 cycles
- Drop constant sensors
- Create rolling window features
- Normalize sensors
"""

import pandas as pd
import numpy as np
from zenml import step
from typing import Tuple


# Constants
RUL_CAP = 125
ROLLING_WINDOW = 5
CONSTANT_SENSORS = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']
KEY_SENSORS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 
               's_13', 's_14', 's_15', 's_17', 's_20', 's_21']


@step
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features for the NASA CMAPSS dataset.
    
    Steps:
    1. Calculate RUL for each time step
    2. Clip RUL at 125 cycles
    3. Drop constant sensors
    4. Create rolling window features (mean, std)
    5. Normalize sensor values
    
    Args:
        df: Raw DataFrame from ingest_data step
        
    Returns:
        Cleaned DataFrame with engineered features
    """
    df = df.copy()
    
    # Step 1: Calculate RUL
    df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    print(f"✓ Calculated RUL (max: {df['RUL'].max()})")
    
    # Step 2: Clip RUL at 125 cycles (industry standard)
    df['RUL_clipped'] = df['RUL'].clip(upper=RUL_CAP)
    print(f"✓ Clipped RUL at {RUL_CAP} cycles")
    
    # Step 3: Drop constant sensors
    sensors_to_drop = [s for s in CONSTANT_SENSORS if s in df.columns]
    df = df.drop(columns=sensors_to_drop)
    print(f"✓ Dropped {len(sensors_to_drop)} constant sensors")
    
    # Step 4: Create rolling window features
    for sensor in KEY_SENSORS:
        if sensor not in df.columns:
            continue
        grouped = df.groupby('unit_nr')[sensor]
        df[f'{sensor}_mean'] = grouped.transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
        )
        df[f'{sensor}_std'] = grouped.transform(
            lambda x: x.rolling(window=ROLLING_WINDOW, min_periods=1).std()
        )
    df = df.fillna(0)
    print(f"✓ Created rolling features for {len(KEY_SENSORS)} sensors")
    
    # Step 5: Normalize sensor values
    for sensor in KEY_SENSORS:
        if sensor not in df.columns:
            continue
        mean_val = df[sensor].mean()
        std_val = df[sensor].std()
        std_val = std_val if std_val > 0 else 1.0
        df[f'{sensor}_norm'] = (df[sensor] - mean_val) / std_val
    print(f"✓ Normalized {len(KEY_SENSORS)} sensors")
    
    print(f"✓ Final shape: {df.shape}")
    
    return df
