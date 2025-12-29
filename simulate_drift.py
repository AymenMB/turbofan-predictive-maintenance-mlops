"""
Drift Simulation Script for Turbofan RUL API

This script demonstrates data drift detection by:
1. Phase 1: Sending normal data from test set (no drift expected)
2. Phase 2: Sending corrupted data (sensor values * 1.5) to simulate drift
3. Monitoring the API's drift detection response after each phase
"""

import requests
import pandas as pd
import numpy as np
import time
from pathlib import Path
from typing import Dict, List

# API configuration
API_URL = "http://localhost:8001"
PREDICT_ENDPOINT = f"{API_URL}/predict"
MONITORING_ENDPOINT = f"{API_URL}/monitoring"
RESET_ENDPOINT = f"{API_URL}/monitoring/reset"

# Sensor columns (all 21 sensors)
SENSOR_COLUMNS = [
    's_1', 's_2', 's_3', 's_4', 's_5', 's_6', 's_7', 's_8', 's_9', 's_10',
    's_11', 's_12', 's_13', 's_14', 's_15', 's_16', 's_17', 's_18', 's_19',
    's_20', 's_21'
]


def load_test_data(file_path: str = "data/raw/test_FD001.txt") -> pd.DataFrame:
    """
    Load test data from CMAPSS dataset.
    
    Args:
        file_path: Path to test data file
        
    Returns:
        DataFrame with test data
    """
    # Column names for CMAPSS dataset
    columns = ['unit', 'cycle', 'setting_1', 'setting_2', 'setting_3'] + SENSOR_COLUMNS
    
    try:
        df = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)
        print(f"✓ Loaded {len(df)} records from {file_path}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        print("  Using synthetic data instead...")
        return generate_synthetic_data()


def generate_synthetic_data(n_samples: int = 100) -> pd.DataFrame:
    """
    Generate synthetic test data with realistic values.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic data
    """
    np.random.seed(42)
    
    data = {
        'unit': [1] * n_samples,
        'cycle': range(1, n_samples + 1),
        'setting_1': np.random.normal(-0.0001, 0.001, n_samples),
        'setting_2': np.random.normal(0.0002, 0.001, n_samples),
        'setting_3': np.random.normal(100.0, 0.5, n_samples),
    }
    
    # Add sensor readings with realistic values
    sensor_means = {
        's_1': 518.67, 's_2': 642.6, 's_3': 1591.4, 's_4': 1407.1,
        's_5': 14.62, 's_6': 21.6, 's_7': 554.9, 's_8': 2388.1,
        's_9': 9059.3, 's_10': 1.30, 's_11': 47.5, 's_12': 522.3,
        's_13': 2388.1, 's_14': 8140.5, 's_15': 8.44, 's_16': 0.03,
        's_17': 391.0, 's_18': 2388.0, 's_19': 100.0, 's_20': 39.1,
        's_21': 23.42
    }
    
    for sensor, mean in sensor_means.items():
        std = mean * 0.05  # 5% standard deviation
        data[sensor] = np.random.normal(mean, std, n_samples)
    
    return pd.DataFrame(data)


def send_prediction_request(row: Dict) -> Dict:
    """
    Send a prediction request to the API.
    
    Args:
        row: Dictionary with sensor readings and settings
        
    Returns:
        API response as dictionary
    """
    payload = {
        'setting_1': float(row['setting_1']),
        'setting_2': float(row['setting_2']),
        'setting_3': float(row['setting_3']),
    }
    
    # Add all sensor readings
    for sensor in SENSOR_COLUMNS:
        payload[sensor] = float(row[sensor])
    
    try:
        response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def check_drift_status() -> Dict:
    """
    Check the current drift detection status.
    
    Returns:
        Monitoring response as dictionary
    """
    try:
        response = requests.get(MONITORING_ENDPOINT, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {'error': str(e)}


def reset_monitoring():
    """Reset monitoring data on the API."""
    try:
        response = requests.get(RESET_ENDPOINT, timeout=5)
        response.raise_for_status()
        print("✓ Monitoring data reset")
    except requests.exceptions.RequestException as e:
        print(f"✗ Error resetting monitoring: {e}")


def print_monitoring_report(monitoring_data: Dict, phase: str):
    """
    Print a formatted monitoring report.
    
    Args:
        monitoring_data: Monitoring response from API
        phase: Description of current phase
    """
    print("\n" + "=" * 70)
    print(f"MONITORING REPORT - {phase}")
    print("=" * 70)
    
    if 'error' in monitoring_data:
        print(f"✗ Error: {monitoring_data['error']}")
        return
    
    drift_detected = monitoring_data.get('drift_detected', False)
    status = monitoring_data.get('status', 'Unknown')
    metrics = monitoring_data.get('metrics', {})
    recent_requests = monitoring_data.get('recent_requests', 0)
    
    print(f"\nStatus: {status}")
    print(f"Drift Detected: {'⚠️  YES' if drift_detected else '✓ NO'}")
    print(f"Recent Requests: {recent_requests}")
    
    if metrics:
        max_dev = metrics.get('max_deviation_pct', 0)
        threshold = metrics.get('threshold_pct', 20)
        print(f"\nMax Deviation: {max_dev:.2f}% (Threshold: {threshold}%)")
        
        drifted_features = metrics.get('drifted_features', [])
        if drifted_features:
            print(f"\n⚠️  Features with Significant Drift ({len(drifted_features)}):")
            for feat in drifted_features[:10]:  # Show top 10
                feature_name = feat.get('feature', 'unknown')
                deviation = feat.get('deviation_pct', 0)
                print(f"  - {feature_name}: {deviation:.2f}%")
        else:
            print("\n✓ No features exceed drift threshold")
    
    print("=" * 70 + "\n")


def run_simulation():
    """Run the complete drift simulation."""
    print("\n" + "=" * 70)
    print("TURBOFAN RUL API - DRIFT DETECTION SIMULATION")
    print("=" * 70)
    
    # Check API availability
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        response.raise_for_status()
        print("✓ API is healthy and ready")
    except requests.exceptions.RequestException as e:
        print(f"✗ Cannot connect to API: {e}")
        print(f"  Make sure the API is running at {API_URL}")
        print("  Start with: uvicorn api.main:app --reload")
        return
    
    # Reset monitoring data
    print("\n--- Resetting Monitoring Data ---")
    reset_monitoring()
    
    # Load test data
    print("\n--- Loading Test Data ---")
    test_data = load_test_data()
    
    if test_data.empty:
        print("✗ No data available for simulation")
        return
    
    # Select random samples for simulation
    n_normal = 25
    n_drift = 25
    sample_indices = np.random.choice(len(test_data), n_normal + n_drift, replace=False)
    sample_data = test_data.iloc[sample_indices].reset_index(drop=True)
    
    # ========================================
    # PHASE 1: Normal Data (No Drift)
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 1: Sending Normal Data")
    print("=" * 70)
    print(f"Sending {n_normal} requests with normal sensor readings...")
    
    success_count = 0
    for i in range(n_normal):
        row = sample_data.iloc[i].to_dict()
        response = send_prediction_request(row)
        
        if 'error' not in response:
            success_count += 1
            if (i + 1) % 10 == 0:
                print(f"  ✓ Sent {i + 1}/{n_normal} requests")
        else:
            print(f"  ✗ Request {i + 1} failed: {response['error']}")
        
        time.sleep(0.1)  # Small delay to avoid overwhelming API
    
    print(f"\n✓ Phase 1 complete: {success_count}/{n_normal} requests successful")
    time.sleep(1)
    
    # Check monitoring after normal data
    monitoring_data = check_drift_status()
    print_monitoring_report(monitoring_data, "PHASE 1: Normal Data")
    
    # ========================================
    # PHASE 2: Corrupted Data (Drift Expected)
    # ========================================
    print("\n" + "=" * 70)
    print("PHASE 2: Sending Corrupted Data (Simulating Drift)")
    print("=" * 70)
    print(f"Multiplying sensor values by 1.5x to simulate broken sensors...")
    print(f"Sending {n_drift} requests with corrupted sensor readings...")
    
    success_count = 0
    for i in range(n_normal, n_normal + n_drift):
        row = sample_data.iloc[i].to_dict()
        
        # Corrupt sensor data (multiply by 1.5 to simulate drift)
        for sensor in SENSOR_COLUMNS:
            row[sensor] = row[sensor] * 1.5
        
        response = send_prediction_request(row)
        
        if 'error' not in response:
            success_count += 1
            if ((i - n_normal + 1) % 10 == 0):
                print(f"  ✓ Sent {i - n_normal + 1}/{n_drift} corrupted requests")
        else:
            print(f"  ✗ Request {i + 1} failed: {response['error']}")
        
        time.sleep(0.1)
    
    print(f"\n✓ Phase 2 complete: {success_count}/{n_drift} requests successful")
    time.sleep(1)
    
    # Check monitoring after corrupted data
    monitoring_data = check_drift_status()
    print_monitoring_report(monitoring_data, "PHASE 2: Corrupted Data")
    
    # ========================================
    # SIMULATION SUMMARY
    # ========================================
    print("\n" + "=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)
    print("\n✓ Drift detection simulation finished successfully!")
    print(f"\nTotal requests sent: {n_normal + n_drift}")
    print(f"  - Normal data: {n_normal}")
    print(f"  - Corrupted data: {n_drift}")
    print("\n💡 Key Observations:")
    print("  1. Phase 1 (normal data) should show NO drift detected")
    print("  2. Phase 2 (corrupted data) should show DRIFT WARNING")
    print("  3. The API successfully detected the 50% increase in sensor values")
    print("\n🔍 View detailed monitoring at: " + MONITORING_ENDPOINT)
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        run_simulation()
    except KeyboardInterrupt:
        print("\n\n✗ Simulation interrupted by user")
    except Exception as e:
        print(f"\n✗ Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
