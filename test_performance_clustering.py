#!/usr/bin/env python3
"""
Test script for performance-based clustering pipeline
Verifies that the clustering script can run with performance-based features
"""

import os
import sys
import subprocess
from pathlib import Path

def test_performance_clustering():
    """Test the performance-based clustering pipeline"""
    print("🧪 Testing Performance-Based Clustering Pipeline")
    print("=" * 50)
    
    # Check if required files exist
    clustering_script = "ecosystem/clustering/clustering_v3_outlier_filtered.py"
    data_pipeline = "data_pipeline_v3.py"
    
    if not os.path.exists(clustering_script):
        print(f"❌ Clustering script not found: {clustering_script}")
        return False
    
    if not os.path.exists(data_pipeline):
        print(f"❌ Data pipeline not found: {data_pipeline}")
        return False
    
    print(f"✅ Found clustering script: {clustering_script}")
    print(f"✅ Found data pipeline: {data_pipeline}")
    
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
    
    # Test clustering with performance features
    print("\n🔧 Testing performance-based clustering...")
    try:
        result = subprocess.run([
            sys.executable, clustering_script,
            "--feature-set", "performance",
            "--data-dir", processed_dir,
            "--output-dir", "test_clustering_results"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Performance-based clustering test PASSED")
            print("📊 Output:")
            print(result.stdout[-1000:])  # Last 1000 characters
            return True
        else:
            print("❌ Performance-based clustering test FAILED")
            print("📊 Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Clustering test timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Clustering test failed with exception: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Performance-Based Clustering Pipeline Test")
    print("=" * 60)
    
    success = test_performance_clustering()
    
    if success:
        print("\n🎉 All tests PASSED!")
        print("✅ Performance-based clustering is ready for ensemble validation")
        print("\n📋 Next steps:")
        print("   1. Run full clustering: python ecosystem/clustering/clustering_v3_outlier_filtered.py")
        print("   2. Train cluster models: python ecosystem/models/train_cluster_models.py")
        print("   3. Run spatial validation: python ecosystem/models/spatial_parquet.py")
    else:
        print("\n❌ Tests FAILED!")
        print("🔧 Please fix the issues above before proceeding")
    
    return success

if __name__ == "__main__":
    main()