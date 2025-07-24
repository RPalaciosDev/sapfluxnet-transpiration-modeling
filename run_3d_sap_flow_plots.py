#!/usr/bin/env python3
"""
Quick 3D Sap Flow Visualization Script
Runs only the 3D interactive plots for faster testing
"""

from sap_flow_visualizations import SapFlowVisualizer
import os
import glob

def main():
    """Run only 3D visualizations"""
    print("🌊 3D Sap Flow Visualization Suite")
    print("=" * 40)
    
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
    
    # Initialize visualizer
    visualizer = SapFlowVisualizer(data_dir=data_dir, output_dir='sap_flow_3d_plots')
    
    # Load data (smaller sample for faster 3D processing)
    data = visualizer.load_sample_sites(n_sites=8)
    
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
    
    # Create only 3D plots
    visualizer.create_all_3d_plots(data)
    
    print(f"\n🎉 3D visualization complete!")
    print(f"🌐 Interactive HTML files created in: sap_flow_3d_plots/")

if __name__ == "__main__":
    main() 