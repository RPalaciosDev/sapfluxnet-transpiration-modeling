#!/usr/bin/env python3
"""
Quick GPU Setup Test for XGBoost
Test script to verify GPU acceleration is working before running full validation
"""

import sys
import subprocess
import numpy as np

def test_nvidia_driver():
    """Test NVIDIA driver and GPU availability"""
    print("🔍 Testing NVIDIA Driver...")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("  ✅ NVIDIA driver working")
            # Extract GPU info
            lines = result.stdout.split('\n')
            for line in lines:
                if any(gpu in line for gpu in ['GeForce', 'Tesla', 'Quadro', 'RTX', 'GTX']):
                    if '|' in line:
                        gpu_info = line.split('|')[1].strip()
                        print(f"  🎮 GPU detected: {gpu_info}")
                        break
            return True
        else:
            print("  ❌ nvidia-smi failed")
            return False
    except Exception as e:
        print(f"  ❌ nvidia-smi error: {e}")
        return False

def test_cuda_toolkit():
    """Test CUDA toolkit installation"""
    print("\n🔍 Testing CUDA Toolkit...")
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Extract CUDA version
            for line in result.stdout.split('\n'):
                if 'release' in line.lower():
                    print(f"  ✅ CUDA toolkit: {line.strip()}")
                    return True
        else:
            print("  ⚠️  nvcc not found (CUDA toolkit may not be installed)")
            return False
    except Exception as e:
        print(f"  ⚠️  nvcc check failed: {e}")
        return False

def test_xgboost_import():
    """Test XGBoost import and version"""
    print("\n🔍 Testing XGBoost...")
    try:
        import xgboost as xgb
        print(f"  ✅ XGBoost version: {xgb.__version__}")
        return xgb
    except ImportError:
        print("  ❌ XGBoost not installed")
        print("  💡 Install with: pip install xgboost")
        return None

def test_xgboost_gpu_support(xgb):
    """Test XGBoost GPU support"""
    print("\n🔍 Testing XGBoost GPU Support...")
    
    # Test 1: Try creating GPU regressor
    try:
        regressor = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0)
        print("  ✅ XGBoost compiled with GPU support")
        del regressor
        gpu_compiled = True
    except Exception as e:
        print(f"  ❌ XGBoost GPU compilation issue: {e}")
        gpu_compiled = False
    
    # Test 2: Try actual GPU training
    if gpu_compiled:
        try:
            print("  🧪 Testing GPU training...")
            
            # Create test data
            X_test = np.random.rand(1000, 10)
            y_test = np.random.rand(1000)
            
            # Create DMatrix and try GPU training
            dtrain = xgb.DMatrix(X_test, label=y_test)
            gpu_params = {
                'objective': 'reg:squarederror',
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'verbosity': 0
            }
            
            model = xgb.train(gpu_params, dtrain, num_boost_round=5, verbose_eval=False)
            print("  🚀 GPU training SUCCESSFUL!")
            
            # Clean up
            del model, dtrain, X_test, y_test
            return True
            
        except Exception as e:
            print(f"  ❌ GPU training failed: {e}")
            return False
    
    return False

def test_pytorch_cuda():
    """Test PyTorch CUDA (optional, for additional verification)"""
    print("\n🔍 Testing PyTorch CUDA (optional)...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✅ PyTorch CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  🔢 CUDA devices: {torch.cuda.device_count()}")
            return True
        else:
            print("  ❌ PyTorch CUDA not available")
            return False
    except ImportError:
        print("  ⚠️  PyTorch not installed (optional)")
        return None

def main():
    """Run all GPU tests"""
    print("🚀 GPU Setup Test for Ecosystem Clustering Pipeline")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: NVIDIA Driver
    if test_nvidia_driver():
        tests_passed += 1
    
    # Test 2: CUDA Toolkit
    if test_cuda_toolkit():
        tests_passed += 1
    
    # Test 3: XGBoost
    xgb = test_xgboost_import()
    if xgb:
        tests_passed += 1
    
    # Test 4: XGBoost GPU
    if xgb and test_xgboost_gpu_support(xgb):
        tests_passed += 1
    
    # Optional: PyTorch CUDA
    test_pytorch_cuda()
    
    # Summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! GPU acceleration should work.")
        print("\n💡 You can now run:")
        print("    python gpu_spatial_validation.py")
    elif tests_passed >= 2:
        print("⚠️  PARTIAL SUCCESS. GPU may work with --force-gpu flag.")
        print("\n💡 Try running:")
        print("    python gpu_spatial_validation.py --force-gpu")
    else:
        print("❌ SETUP ISSUES DETECTED. Please fix the following:")
        if tests_passed == 0:
            print("  - Install NVIDIA drivers")
            print("  - Install CUDA toolkit")
            print("  - Install XGBoost with GPU support")
        print("\n📚 See GPU_TROUBLESHOOTING.md for detailed solutions.")
    
    print("\n🔧 Quick fixes to try:")
    print("  pip install xgboost[gpu]")
    print("  conda install -c conda-forge xgboost-gpu cudatoolkit")

if __name__ == "__main__":
    main()