#!/usr/bin/env python3
"""
Model-Integrated Sap Flow Visualization Script
Runs visualizations that combine trained XGBoost models with sap flow data analysis
"""

from sap_flow_visualizations import SapFlowVisualizer
import os
import glob

def main():
    """Run model-integrated visualizations"""
    print("🤖 Model-Integrated Sap Flow Visualization Suite")
    print("=" * 50)
    
    # Check for data directory
    data_dirs = ['processed_parquet', '../processed_parquet', 'ecosystem/processed_parquet']
    data_dir = None
    
    for dir_path in data_dirs:
        if os.path.exists(dir_path) and glob.glob(os.path.join(dir_path, "*.parquet")):
            data_dir = dir_path
            break
    
    if not data_dir:
        print("❌ No parquet data directory found!")
        print("Expected locations: processed_parquet/, ../processed_parquet/, ecosystem/processed_parquet/")
        return
    
    # Check for models directory
    models_dirs = [
        'ecosystem/models/results/cluster_models',
        '../ecosystem/models/results/cluster_models',
        './results/cluster_models'
    ]
    models_dir = None
    
    for dir_path in models_dirs:
        if os.path.exists(dir_path) and glob.glob(os.path.join(dir_path, "xgb_model_cluster_*.json")):
            models_dir = dir_path
            break
    
    if not models_dir:
        print("❌ No trained models directory found!")
        print("Expected locations:")
        for dir_path in models_dirs:
            print(f"  - {dir_path}")
        print("\n💡 Train models first using:")
        print("   cd ecosystem/models && python train_cluster_models.py")
        return
    
    # Initialize visualizer with model integration
    visualizer = SapFlowVisualizer(
        data_dir=data_dir, 
        output_dir='sap_flow_model_plots',
        models_dir=models_dir
    )
    
    # Check if models were loaded
    if not visualizer.models:
        print("❌ No models were loaded successfully")
        return
    
    # Load data (smaller sample for model testing)
    data = visualizer.load_sample_sites(n_sites=10)
    
    if data is None:
        print("❌ No data loaded, exiting...")
        return
    
    # Filter data
    initial_len = len(data)
    data = data[
        (data['sap_flow'].notna()) & 
        (data['sap_flow'] >= 0) & 
        (data['sap_flow'] < data['sap_flow'].quantile(0.999))
    ]
    print(f"🧹 Filtered data: {len(data):,} records ({initial_len - len(data):,} removed)")
    
    # Create model-integrated visualizations
    visualizer.create_all_model_visualizations(data)
    
    print(f"\n🎉 Model-integrated visualization complete!")
    print(f"🤖 Interactive model plots created in: sap_flow_model_plots/")
    print(f"\n💡 Generated visualizations show:")
    print(f"   - Model vs observed comparisons")
    print(f"   - Prediction time series analysis")
    print(f"   - 3D model predictions in environmental space")
    print(f"   - 3D feature importance across clusters")

if __name__ == "__main__":
    main() 