"""
Data ingestion step for ZenML pipeline.

Loads raw CMAPSS turbofan data.
"""

import pandas as pd
from zenml import step


@step
def ingest_data(data_path: str = "data/raw/train_FD001.txt") -> pd.DataFrame:
    """
    Load raw NASA CMAPSS FD001 training data.
    
    Args:
        data_path: Path to the raw training data file
        
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
    
    print(f"✓ Ingested {len(df)} rows from {data_path}")
    print(f"✓ Dataset contains {df['unit_nr'].nunique()} unique engines")
    
    return df
