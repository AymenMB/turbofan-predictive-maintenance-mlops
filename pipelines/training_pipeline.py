"""
Training pipeline for Turbofan RUL prediction.

Orchestrates data ingestion, cleaning, training, and evaluation.
"""

from zenml import pipeline

from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model


@pipeline
def training_pipeline(
    data_path: str = "data/raw/train_FD001.txt",
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6
):
    """
    End-to-end training pipeline for RUL prediction.
    
    Pipeline steps:
    1. Ingest raw data
    2. Clean data and calculate RUL
    3. Train XGBoost model
    4. Evaluate model performance
    
    Args:
        data_path: Path to raw training data
        n_estimators: Number of XGBoost estimators
        learning_rate: XGBoost learning rate
        max_depth: Maximum tree depth
    """
    # Step 1: Ingest raw data
    raw_data = ingest_data(data_path=data_path)
    
    # Step 2: Clean data and calculate RUL
    cleaned_data = clean_data(df=raw_data)
    
    # Step 3: Train model
    model = train_model(
        df=cleaned_data,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth
    )
    
    # Step 4: Evaluate model
    metrics = evaluate_model(
        model=model,
        df=cleaned_data
    )
    
    return metrics
