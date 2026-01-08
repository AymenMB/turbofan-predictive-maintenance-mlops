"""
Data preprocessing module for NASA CMAPSS FD001 dataset.

This module handles loading, cleaning, and feature engineering for the turbofan
engine degradation dataset. Implements the same feature engineering as the R
implementation to achieve competitive RMSE (~17-20 cycles).

Key Features:
- Rolling window statistics (mean, std) to capture degradation trends
- RUL clipping at 125 cycles (industry standard)
- Sensor normalization for improved model convergence
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


# Constants
RUL_CAP = 125  # Maximum RUL value (industry standard)
ROLLING_WINDOW = 5  # Window size for rolling statistics

# Sensors with constant/near-zero variance (no information for prediction)
CONSTANT_SENSORS = ['s_1', 's_5', 's_10', 's_16', 's_18', 's_19']

# Most important sensors based on R analysis (Cox PH, Random Forest importance)
KEY_SENSORS = ['s_2', 's_3', 's_4', 's_7', 's_8', 's_9', 's_11', 's_12', 
               's_13', 's_14', 's_15', 's_17', 's_20', 's_21']


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw NASA CMAPSS data from text file.
    
    Args:
        data_path: Path to the training data file
        
    Returns:
        Raw DataFrame with named columns
    """
    # Define column names for the dataset
    # 1 unit ID + 1 time + 3 operational settings + 21 sensor measurements
    column_names = ['unit_nr', 'time_cycles'] + \
                   [f'setting_{i}' for i in range(1, 4)] + \
                   [f's_{i}' for i in range(1, 22)]
    
    # Load the data
    df = pd.read_csv(data_path, sep=r'\s+', header=None, names=column_names)
    
    return df


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each time step.
    
    RUL = max_cycle_for_unit - current_cycle
    
    Args:
        df: DataFrame with unit_nr and time_cycles columns
        
    Returns:
        DataFrame with RUL column added
    """
    df = df.copy()
    df['RUL'] = df.groupby('unit_nr')['time_cycles'].transform('max') - df['time_cycles']
    return df


def clip_rul(df: pd.DataFrame, cap: int = RUL_CAP) -> pd.DataFrame:
    """
    Clip RUL values to a maximum value.
    
    Industry standard: Cap at 125 cycles because sensors are essentially
    flat (healthy engine) when RUL > 125, adding noise to training.
    
    Args:
        df: DataFrame with RUL column
        cap: Maximum RUL value (default: 125)
        
    Returns:
        DataFrame with RUL_clipped column added
    """
    df = df.copy()
    df['RUL_clipped'] = df['RUL'].clip(upper=cap)
    return df


def create_rolling_features(df: pd.DataFrame, 
                           sensors: List[str],
                           window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Create rolling window features for time-series sensor data.
    
    For each sensor, compute:
    - Rolling mean: Captures the degradation trend
    - Rolling std: Captures instability/volatility before failure
    
    Args:
        df: DataFrame with sensor columns
        sensors: List of sensor column names
        window: Rolling window size (default: 5)
        
    Returns:
        DataFrame with rolling features added
    """
    df = df.copy()
    
    for sensor in sensors:
        if sensor not in df.columns:
            continue
            
        # Create rolling features grouped by unit (engine)
        grouped = df.groupby('unit_nr')[sensor]
        
        # Rolling mean - captures degradation trend
        df[f'{sensor}_mean'] = grouped.transform(
            lambda x: x.rolling(window=window, min_periods=1).mean()
        )
        
        # Rolling std - captures instability before failure
        df[f'{sensor}_std'] = grouped.transform(
            lambda x: x.rolling(window=window, min_periods=1).std()
        )
    
    # Fill NaN values from rolling operations
    df = df.fillna(0)
    
    return df


def normalize_sensors(df: pd.DataFrame, 
                     sensors: List[str],
                     fit: bool = True,
                     stats: Optional[dict] = None) -> Tuple[pd.DataFrame, dict]:
    """
    Z-score normalize sensor values.
    
    Normalization improves model convergence and makes features comparable.
    
    Args:
        df: DataFrame with sensor columns
        sensors: List of sensor column names to normalize
        fit: If True, compute stats from data. If False, use provided stats.
        stats: Pre-computed mean/std statistics (for inference)
        
    Returns:
        Tuple of (normalized DataFrame, statistics dict)
    """
    df = df.copy()
    
    if stats is None:
        stats = {}
    
    for sensor in sensors:
        if sensor not in df.columns:
            continue
            
        if fit:
            # Compute and store statistics
            mean_val = df[sensor].mean()
            std_val = df[sensor].std()
            # Avoid division by zero
            std_val = std_val if std_val > 0 else 1.0
            stats[sensor] = {'mean': mean_val, 'std': std_val}
        else:
            # Use provided statistics
            if sensor not in stats:
                continue
            mean_val = stats[sensor]['mean']
            std_val = stats[sensor]['std']
        
        # Apply normalization
        df[f'{sensor}_norm'] = (df[sensor] - mean_val) / std_val
    
    return df, stats


def load_and_process_data(data_path: str = "data/raw/train_FD001.txt",
                          add_rolling: bool = True,
                          add_normalization: bool = True,
                          clip_rul_values: bool = True) -> pd.DataFrame:
    """
    Load and preprocess NASA CMAPSS FD001 training data with advanced features.
    
    Complete preprocessing pipeline:
    1. Load space-separated data without headers
    2. Calculate Remaining Useful Life (RUL)
    3. Clip RUL at 125 cycles (industry standard)
    4. Drop sensors with constant values (zero variance)
    5. Create rolling window features (mean, std)
    6. Normalize sensor values
    
    Args:
        data_path: Path to the raw training data file
        add_rolling: Whether to add rolling window features
        add_normalization: Whether to normalize sensors
        clip_rul_values: Whether to clip RUL at 125
        
    Returns:
        Cleaned DataFrame with engineered features
    """
    print(f"Loading data from {data_path}...")
    
    # Step 1: Load raw data
    df = load_raw_data(data_path)
    print(f"✓ Loaded {len(df)} rows, {df['unit_nr'].nunique()} engines")
    
    # Step 2: Calculate RUL
    df = calculate_rul(df)
    print(f"✓ Calculated RUL (max: {df['RUL'].max()}, min: {df['RUL'].min()})")
    
    # Step 3: Clip RUL at 125 (industry standard)
    if clip_rul_values:
        df = clip_rul(df, cap=RUL_CAP)
        print(f"✓ Clipped RUL at {RUL_CAP} cycles")
    
    # Step 4: Drop constant sensors
    df = df.drop(columns=[s for s in CONSTANT_SENSORS if s in df.columns])
    print(f"✓ Dropped {len(CONSTANT_SENSORS)} constant sensors")
    
    # Step 5: Create rolling window features for key sensors
    if add_rolling:
        df = create_rolling_features(df, KEY_SENSORS, window=ROLLING_WINDOW)
        print(f"✓ Created rolling features (window={ROLLING_WINDOW}) for {len(KEY_SENSORS)} sensors")
    
    # Step 6: Normalize sensor values
    if add_normalization:
        # Normalize only raw sensors, not rolling features (they're already derived)
        sensors_to_normalize = [s for s in KEY_SENSORS if s in df.columns]
        df, _ = normalize_sensors(df, sensors_to_normalize, fit=True)
        print(f"✓ Normalized {len(sensors_to_normalize)} sensors")
    
    print(f"✓ Final shape: {df.shape}")
    print(f"✓ Features: {len([c for c in df.columns if c not in ['unit_nr', 'time_cycles', 'RUL', 'RUL_clipped']])} columns")
    
    return df


def get_feature_columns(df: pd.DataFrame, 
                        include_rolling: bool = True,
                        include_normalized: bool = True,
                        include_raw: bool = False) -> List[str]:
    """
    Get list of feature columns for model training.
    
    Args:
        df: DataFrame with all columns
        include_rolling: Include rolling mean/std features
        include_normalized: Include normalized sensor features
        include_raw: Include raw sensor values
        
    Returns:
        List of feature column names
    """
    exclude_cols = ['unit_nr', 'time_cycles', 'RUL', 'RUL_clipped']
    
    feature_cols = []
    
    for col in df.columns:
        if col in exclude_cols:
            continue
            
        is_rolling = col.endswith('_mean') or col.endswith('_std')
        is_normalized = col.endswith('_norm')
        is_setting = col.startswith('setting_')
        is_raw = not is_rolling and not is_normalized and not is_setting
        
        # Always include settings
        if is_setting:
            feature_cols.append(col)
            continue
            
        if is_rolling and include_rolling:
            feature_cols.append(col)
        elif is_normalized and include_normalized:
            feature_cols.append(col)
        elif is_raw and include_raw:
            feature_cols.append(col)
    
    return sorted(feature_cols)


if __name__ == "__main__":
    # Test the preprocessing function
    print("=" * 70)
    print("Testing NASA CMAPSS Data Preprocessing with Feature Engineering")
    print("=" * 70)
    print()
    
    # Load and process the data
    df = load_and_process_data()
    
    print("\n" + "=" * 70)
    print("Feature Summary")
    print("=" * 70)
    
    # Get different feature sets
    rolling_features = [c for c in df.columns if c.endswith('_mean') or c.endswith('_std')]
    norm_features = [c for c in df.columns if c.endswith('_norm')]
    
    print(f"\nRolling features ({len(rolling_features)}):")
    print(f"  {rolling_features[:6]}... (showing first 6)")
    
    print(f"\nNormalized features ({len(norm_features)}):")
    print(f"  {norm_features[:6]}... (showing first 6)")
    
    print("\n" + "=" * 70)
    print("RUL Statistics (Clipped)")
    print("=" * 70)
    print(df['RUL_clipped'].describe())
    
    print("\n" + "=" * 70)
    print("Sample Feature Values (Engine 1, Last 5 Cycles)")
    print("=" * 70)
    sample = df[df['unit_nr'] == 1].tail(5)[['time_cycles', 'RUL', 'RUL_clipped', 
                                              's_11_mean', 's_11_std', 's_11_norm']]
    print(sample.to_string())
