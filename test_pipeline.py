"""Simple test script for the ZenML pipeline."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("Testing pipeline import...")
try:
    from pipelines.training_pipeline import training_pipeline
    print("✓ Pipeline imported successfully!")
except Exception as e:
    print(f"✗ Error importing pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting pipeline execution...")
try:
    metrics = training_pipeline(
        data_path="data/raw/train_FD001.txt",
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6
    )
    print(f"\n✓ Pipeline completed successfully!")
    print(f"Metrics: {metrics}")
except Exception as e:
    print(f"\n✗ Error running pipeline: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
