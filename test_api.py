"""
Test script for the Turbofan RUL Prediction API.

This script tests the API endpoints locally (without Docker).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
import json


def test_health_endpoint(base_url="http://localhost:8000"):
    """Test the health check endpoint."""
    print("=" * 70)
    print("Testing /health endpoint...")
    print("=" * 70)
    
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✓ Health check passed!")
            return True
        else:
            print("✗ Health check failed!")
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure it's running!")
        print("  Run: uvicorn api.main:app --reload")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_predict_endpoint(base_url="http://localhost:8000"):
    """Test the prediction endpoint with sample data."""
    print("\n" + "=" * 70)
    print("Testing /predict endpoint...")
    print("=" * 70)
    
    # Sample engine data (realistic values from training set)
    sample_data = {
        "setting_1": -0.0007,
        "setting_2": -0.0004,
        "setting_3": 100.0,
        "s_1": 518.67,
        "s_2": 641.82,
        "s_3": 1589.70,
        "s_4": 1400.60,
        "s_5": 14.62,
        "s_6": 21.61,
        "s_7": 554.36,
        "s_8": 2388.06,
        "s_9": 9046.19,
        "s_10": 1.30,
        "s_11": 47.47,
        "s_12": 521.66,
        "s_13": 2388.02,
        "s_14": 8138.62,
        "s_15": 8.4195,
        "s_16": 0.03,
        "s_17": 392,
        "s_18": 2388,
        "s_19": 100.0,
        "s_20": 39.06,
        "s_21": 23.4190
    }
    
    try:
        print("Sending prediction request...")
        print(f"Input data: setting_1={sample_data['setting_1']}, setting_2={sample_data['setting_2']}, ...")
        
        response = requests.post(
            f"{base_url}/predict",
            json=sample_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Prediction successful!")
            print(f"  RUL: {result['RUL']} cycles")
            print(f"  Status: {result['status']}")
            print(f"  Confidence: {result['confidence']}")
            return True
        else:
            print(f"✗ Prediction failed!")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API. Make sure it's running!")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_model_info_endpoint(base_url="http://localhost:8000"):
    """Test the model info endpoint."""
    print("\n" + "=" * 70)
    print("Testing /model-info endpoint...")
    print("=" * 70)
    
    try:
        response = requests.get(f"{base_url}/model-info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            info = response.json()
            print(f"\n✓ Model info retrieved!")
            print(f"  Model Type: {info['model_type']}")
            print(f"  Test RMSE: {info['performance']['test_rmse']} cycles")
            print(f"  Optimization: {info['optimization']}")
            print(f"  Features: {info['features']['after_preprocessing']}")
            return True
        else:
            print("✗ Failed to retrieve model info!")
            return False
            
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to API.")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_multiple_predictions(base_url="http://localhost:8000", n=3):
    """Test multiple predictions with different inputs."""
    print("\n" + "=" * 70)
    print(f"Testing {n} different predictions...")
    print("=" * 70)
    
    # Different scenarios
    test_cases = [
        {
            "name": "Early lifecycle (high RUL expected)",
            "data": {
                "setting_1": -0.0007, "setting_2": -0.0004, "setting_3": 100.0,
                "s_1": 518.67, "s_2": 642.0, "s_3": 1590.0, "s_4": 1400.0,
                "s_5": 14.62, "s_6": 21.6, "s_7": 554.0, "s_8": 2388.0,
                "s_9": 9046.0, "s_10": 1.30, "s_11": 47.5, "s_12": 522.0,
                "s_13": 2388.0, "s_14": 8139.0, "s_15": 8.42, "s_16": 0.03,
                "s_17": 392, "s_18": 2388, "s_19": 100.0, "s_20": 39.0, "s_21": 23.4
            }
        },
        {
            "name": "Mid lifecycle (moderate RUL)",
            "data": {
                "setting_1": 0.0023, "setting_2": 0.0003, "setting_3": 100.0,
                "s_1": 518.67, "s_2": 643.0, "s_3": 1592.0, "s_4": 1407.0,
                "s_5": 14.62, "s_6": 21.6, "s_7": 555.0, "s_8": 2389.0,
                "s_9": 9050.0, "s_10": 1.30, "s_11": 47.8, "s_12": 523.0,
                "s_13": 2389.0, "s_14": 8150.0, "s_15": 8.44, "s_16": 0.03,
                "s_17": 393, "s_18": 2389, "s_19": 100.0, "s_20": 39.2, "s_21": 23.5
            }
        },
        {
            "name": "Late lifecycle (low RUL expected)",
            "data": {
                "setting_1": 0.0042, "setting_2": 0.0006, "setting_3": 100.0,
                "s_1": 518.67, "s_2": 644.5, "s_3": 1595.0, "s_4": 1420.0,
                "s_5": 14.62, "s_6": 21.7, "s_7": 557.0, "s_8": 2391.0,
                "s_9": 9060.0, "s_10": 1.30, "s_11": 48.2, "s_12": 525.0,
                "s_13": 2391.0, "s_14": 8170.0, "s_15": 8.47, "s_16": 0.03,
                "s_17": 395, "s_18": 2391, "s_19": 100.0, "s_20": 39.5, "s_21": 23.7
            }
        }
    ]
    
    results = []
    for i, test_case in enumerate(test_cases[:n], 1):
        print(f"\nTest {i}: {test_case['name']}")
        try:
            response = requests.post(
                f"{base_url}/predict",
                json=test_case['data']
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ RUL: {result['RUL']} cycles | Status: {result['status']}")
                results.append(True)
            else:
                print(f"  ✗ Failed (Status {response.status_code})")
                results.append(False)
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append(False)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\nSuccess rate: {success_rate:.0f}% ({sum(results)}/{len(results)})")
    return all(results)


def main():
    """Run all API tests."""
    print("\n" + "=" * 70)
    print("TURBOFAN RUL PREDICTION API - TEST SUITE")
    print("=" * 70)
    print("\nMake sure the API is running:")
    print("  Option 1: uvicorn api.main:app --reload")
    print("  Option 2: python -m api.main")
    print()
    
    base_url = "http://localhost:8000"
    
    # Run tests
    test_results = {
        "Health Check": test_health_endpoint(base_url),
        "Model Info": test_model_info_endpoint(base_url),
        "Single Prediction": test_predict_endpoint(base_url),
        "Multiple Predictions": test_multiple_predictions(base_url, n=3)
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    total_passed = sum(test_results.values())
    total_tests = len(test_results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if all(test_results.values()):
        print("\n🎉 All tests passed! API is working correctly.")
        return 0
    else:
        print("\n⚠ Some tests failed. Please check the API.")
        return 1


if __name__ == "__main__":
    exit(main())
