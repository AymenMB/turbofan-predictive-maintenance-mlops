"""
Run script for Turbofan RUL prediction ZenML pipeline.

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pipelines.training_pipeline import training_pipeline


def main():
    """
    Initialize and run the training pipeline.
    """
    print("=" * 70)
    print("TURBOFAN RUL PREDICTION - ZenML PIPELINE")
    print("=" * 70)
    print("\nInitializing ZenML pipeline...")
    
    # Run the pipeline with default parameters
    # You can modify these parameters for experimentation
    pipeline_run = training_pipeline(
        data_path="data/raw/train_FD001.txt",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    
    print("\n" + "=" * 70)
    print("✓ PIPELINE EXECUTION COMPLETE")
    print("=" * 70)
    print("\nTo view pipeline runs:")
    print("  zenml up  # Start ZenML dashboard")
    print("  # Then open http://localhost:8237")
    print("\nTo view experiment metrics:")
    print("  .venv\\Scripts\\mlflow.exe ui")
    print("  # Then open http://localhost:5000")


if __name__ == "__main__":
    main()
