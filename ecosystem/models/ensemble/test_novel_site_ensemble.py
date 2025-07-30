#!/usr/bin/env python3
"""
Test script for novel site ensemble testing pipeline
Verifies that the ensemble testing script can run with trained models
"""

import os
import sys
import subprocess
from pathlib import Path

def test_novel_site_ensemble():
    """Test the novel site ensemble testing pipeline"""
    print("🧪 Testing Novel Site Ensemble Pipeline")
    print("=" * 50)
    
    # Check if required files exist
    ensemble_script = "ecosystem/models/novel_site_ensemble_testing.py"
    clustering_script = "ecosystem/clustering/clustering_v3_outlier_filtered.py"
    train_script = "ecosystem/models/train_cluster_models.py"
    
    if not os.path.exists(ensemble_script):
        print(f"❌ Ensemble script not found: {ensemble_script}")
        return False
    
    if not os.path.exists(clustering_script):
        print(f"❌ Clustering script not found: {clustering_script}")
        return False
    
    if not os.path.exists(train_script):
        print(f"❌ Training script not found: {train_script}")
        return False
    
    print(f"✅ Found ensemble script: {ensemble_script}")
    print(f"✅ Found clustering script: {clustering_script}")
    print(f"✅ Found training script: {train_script}")
    
    # Check if processed data exists
    processed_dir = "processed_parquet"
    if not os.path.exists(processed_dir):
        print(f"⚠️  Processed data directory not found: {processed_dir}")
        print("💡 You may need to run the data pipeline first:")
        print("   python data_pipeline_v3.py --clean-mode --export-format parquet")
        return False
    
    parquet_files = [f for f in os.listdir(processed_dir) if f.endswith('.parquet')]
    if not parquet_files:
        print(f"❌ No parquet files found in {processed_dir}")
        print("💡 You need to run the data pipeline first:")
        print("   python data_pipeline_v3.py --clean-mode --export-format parquet")
        return False
    
    print(f"✅ Found {len(parquet_files)} parquet files in {processed_dir}")
    
    # Check if clustering results exist
    clustering_results = "ecosystem/evaluation/clustering_results"
    if not os.path.exists(clustering_results):
        print(f"⚠️  Clustering results not found: {clustering_results}")
        print("💡 You may need to run clustering first:")
        print("   python ecosystem/clustering/clustering_v3_outlier_filtered.py --feature-set performance")
        return False
    
    cluster_files = [f for f in os.listdir(clustering_results) if f.startswith('advanced_site_clusters_')]
    if not cluster_files:
        print(f"❌ No clustering files found in {clustering_results}")
        print("💡 You need to run clustering first:")
        print("   python ecosystem/clustering/clustering_v3_outlier_filtered.py --feature-set performance")
        return False
    
    print(f"✅ Found {len(cluster_files)} clustering files")
    
    # Check if trained models exist
    models_dir = "ecosystem/models/results/cluster_models"
    if not os.path.exists(models_dir):
        print(f"⚠️  Trained models not found: {models_dir}")
        print("💡 You may need to train models first:")
        print("   python ecosystem/models/train_cluster_models.py")
        return False
    
    model_files = [f for f in os.listdir(models_dir) if f.startswith('cluster_model_') and f.endswith('.json')]
    if not model_files:
        print(f"❌ No trained models found in {models_dir}")
        print("💡 You need to train models first:")
        print("   python ecosystem/models/train_cluster_models.py")
        return False
    
    print(f"✅ Found {len(model_files)} trained models")
    
    # Test novel site ensemble testing
    print("\n🔧 Testing novel site ensemble testing...")
    try:
        result = subprocess.run([
            sys.executable, ensemble_script,
            "--test-fraction", "0.1",  # Use small fraction for testing
            "--output-dir", "test_novel_ensemble_results"
        ], capture_output=True, text=True, timeout=600)  # 10 minute timeout
        
        if result.returncode == 0:
            print("✅ Novel site ensemble testing PASSED")
            print("📊 Output:")
            print(result.stdout[-1000:])  # Last 1000 characters
            return True
        else:
            print("❌ Novel site ensemble testing FAILED")
            print("📊 Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Ensemble testing timed out (10 minutes)")
        return False
    except Exception as e:
        print(f"❌ Ensemble testing failed with exception: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Novel Site Ensemble Testing Pipeline Test")
    print("=" * 60)
    
    success = test_novel_site_ensemble()
    
    if success:
        print("\n🎉 All tests PASSED!")
        print("✅ Novel site ensemble testing is ready")
        print("\n📋 Next steps:")
        print("   1. Run full ensemble testing: python ecosystem/models/novel_site_ensemble_testing.py")
        print("   2. Analyze results and compare weighting strategies")
        print("   3. Proceed to cross-ecosystem validation")
    else:
        print("\n❌ Tests FAILED!")
        print("🔧 Please fix the issues above before proceeding")
    
    return success

if __name__ == "__main__":
    main() 