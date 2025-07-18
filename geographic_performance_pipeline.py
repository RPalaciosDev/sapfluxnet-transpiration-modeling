#!/usr/bin/env python3
"""
Geographic Performance Pipeline for SAPFLUXNET Validation Results
================================================================

This pipeline processes results from all validation runs and creates
GeoJSON files for geographic comparison of model performance.

Supports:
- Random split validation
- Spatial validation (multiple variants)
- Temporal validation (multiple variants)
- Cross-validation results

Author: AI Assistant
Date: 2025-01-17
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import os
import json
import glob
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

class GeographicPerformancePipeline:
    """Pipeline for creating geographic visualizations of model performance."""
    
    def __init__(self, project_path):
        """Initialize the pipeline with project path."""
        self.project_path = Path(project_path)
        self.spatial_data = None
        self.output_dir = self.project_path / "geographic_analysis"
        self.output_dir.mkdir(exist_ok=True)
        
    def load_spatial_data(self):
        """Load the spatial reference data."""
        print("Loading spatial reference data...")
        
        shapefile_path = self.project_path / "shapefiles" / "sapfluxnet_sites.shp"
        if not shapefile_path.exists():
            print(f"Error: Spatial data not found at {shapefile_path}")
            return False
        
        try:
            self.spatial_data = gpd.read_file(shapefile_path)
            print(f"Loaded {len(self.spatial_data)} spatial features")
            return True
        except Exception as e:
            print(f"Error loading spatial data: {e}")
            return False
    
    def discover_validation_runs(self):
        """Discover all validation runs in the project."""
        print("\nDiscovering validation runs...")
        
        validation_runs = []
        models_dir = self.project_path / "xgboost_scripts" / "external_memory_models"
        
        if not models_dir.exists():
            print(f"Error: Models directory not found at {models_dir}")
            return []
        
        # Look for different validation types
        validation_types = [
            "random_split",
            "spatial_validation",
            "spatial_validation_no_proxies", 
            "spatial_validation_universal_features",
            "temporal_validation",
            "temporal_validation_chronological",
            "temporal_validation_chronological_nonspecific",
            "temporal_validation_proper"
        ]
        
        for val_type in validation_types:
            val_dir = models_dir / val_type
            if val_dir.exists():
                # Look for site-specific results
                site_files = list(val_dir.glob("*sites*.csv"))
                if site_files:
                    for site_file in site_files:
                        validation_runs.append({
                            'type': val_type,
                            'path': site_file,
                            'name': f"{val_type}_{site_file.stem}"
                        })
                        print(f"Found: {val_type} - {site_file.name}")
                else:
                    # Check if it's a random split (might not have site-specific results)
                    if val_type == "random_split":
                        validation_runs.append({
                            'type': val_type,
                            'path': val_dir,
                            'name': val_type,
                            'is_random_split': True
                        })
                        print(f"Found: {val_type} (random split)")
        
        print(f"Discovered {len(validation_runs)} validation runs")
        return validation_runs
    
    def process_site_specific_results(self, run_info):
        """Process site-specific validation results."""
        print(f"\nProcessing {run_info['name']}...")
        
        try:
            # Load performance data
            performance_df = pd.read_csv(run_info['path'])
            print(f"  Loaded {len(performance_df)} site records")
            
            # Basic validation
            required_fields = ['site', 'test_r2', 'test_rmse']
            missing_fields = [f for f in required_fields if f not in performance_df.columns]
            if missing_fields:
                print(f"  Warning: Missing fields {missing_fields}")
                return None
            
            # Add performance categories
            performance_df['category'] = performance_df['test_r2'].apply(self._categorize_performance)
            
            # Merge with spatial data
            merged_data = self._merge_with_spatial_data(performance_df)
            if merged_data is None:
                return None
            
            # Add metadata
            merged_data.attrs['validation_type'] = run_info['type']
            merged_data.attrs['run_name'] = run_info['name']
            merged_data.attrs['timestamp'] = datetime.now().isoformat()
            merged_data.attrs['total_sites'] = len(merged_data)
            
            return merged_data
            
        except Exception as e:
            print(f"  Error processing {run_info['name']}: {e}")
            return None
    
    def process_random_split_results(self, run_info):
        """Process random split results (no site-specific data)."""
        print(f"\nProcessing {run_info['name']} (random split)...")
        
        try:
            # For random split, we need to create a summary
            # Load the main results file
            results_file = run_info['path'] / "sapfluxnet_external_memory_metrics.txt"
            if not results_file.exists():
                print(f"  Error: Metrics file not found at {results_file}")
                return None
            
            # Read metrics
            with open(results_file, 'r') as f:
                metrics_content = f.read()
            
            # Extract overall metrics
            metrics = self._parse_metrics_file(metrics_content)
            
            # Create a summary GeoJSON with overall performance
            summary_data = self._create_random_split_summary(metrics, run_info)
            
            return summary_data
            
        except Exception as e:
            print(f"  Error processing random split: {e}")
            return None
    
    def _parse_metrics_file(self, content):
        """Parse metrics file to extract performance values."""
        metrics = {}
        
        # Look for R² and RMSE values
        lines = content.split('\n')
        for line in lines:
            if 'R²' in line or 'R2' in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    metrics['test_r2'] = value
                except:
                    pass
            elif 'RMSE' in line:
                try:
                    value = float(line.split(':')[-1].strip())
                    metrics['test_rmse'] = value
                except:
                    pass
        
        return metrics
    
    def _create_random_split_summary(self, metrics, run_info):
        """Create a summary for random split results."""
        # Create a simple summary with overall performance
        summary = {
            'validation_type': run_info['type'],
            'run_name': run_info['name'],
            'overall_r2': metrics.get('test_r2', 0),
            'overall_rmse': metrics.get('test_rmse', 0),
            'note': 'Random split - no site-specific results available'
        }
        
        # Save summary
        summary_file = self.output_dir / f"{run_info['name']}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Created summary: {summary_file}")
        return summary
    
    def _categorize_performance(self, r2):
        """Categorize performance based on R² value."""
        if r2 >= 0.7:
            return "Excellent"
        elif r2 >= 0.5:
            return "Good"
        elif r2 >= 0.3:
            return "Fair"
        elif r2 >= 0:
            return "Poor"
        else:
            return "Very Poor"
    
    def _merge_with_spatial_data(self, performance_df):
        """Merge performance data with spatial data."""
        if self.spatial_data is None:
            print("  Error: Spatial data not loaded")
            return None
        
        # Clean site names for matching
        performance_df['site_clean'] = performance_df['site'].str.strip()
        self.spatial_data['site_clean'] = self.spatial_data['site'].astype(str).str.strip()
        
        # Merge data
        merged_data = self.spatial_data.merge(
            performance_df, 
            left_on='site_clean', 
            right_on='site_clean', 
            how='inner'
        )
        
        if len(merged_data) == 0:
            print("  Warning: No sites matched between performance and spatial data")
            return None
        
        print(f"  Successfully merged {len(merged_data)} records")
        
        # Add derived fields
        merged_data = self._add_derived_fields(merged_data)
        
        return merged_data
    
    def _add_derived_fields(self, data):
        """Add derived fields for analysis."""
        # Normalize R² values
        r2_min, r2_max = data['test_r2'].min(), data['test_r2'].max()
        if r2_max > r2_min:
            data['r2_normalized'] = (data['test_r2'] - r2_min) / (r2_max - r2_min)
        else:
            data['r2_normalized'] = 0.5
        
        # Normalize RMSE values
        rmse_min, rmse_max = data['test_rmse'].min(), data['test_rmse'].max()
        if rmse_max > rmse_min:
            data['rmse_normalized'] = (data['test_rmse'] - rmse_min) / (rmse_max - rmse_min)
        else:
            data['rmse_normalized'] = 0.5
        
        # Performance score
        data['performance_score'] = data['r2_normalized'] * (1 - data['rmse_normalized'])
        
        # Size categories
        if 'test_samples' in data.columns:
            data['size_category'] = pd.cut(
                data['test_samples'], 
                bins=[0, 1000, 2000, 3000, float('inf')], 
                labels=['Small', 'Medium', 'Large', 'Very Large']
            )
        
        return data
    
    def export_geojson(self, data, run_name):
        """Export data as GeoJSON."""
        if data is None:
            return None
        
        # Create output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{run_name}_{timestamp}.geojson"
        output_path = self.output_dir / filename
        
        try:
            # Export as GeoJSON
            data.to_file(output_path, driver='GeoJSON')
            print(f"  ✓ GeoJSON exported: {output_path}")
            
            # Also export as CSV for compatibility
            csv_path = output_path.with_suffix('.csv')
            data.to_csv(csv_path, index=False)
            print(f"  ✓ CSV exported: {csv_path}")
            
            return output_path
            
        except Exception as e:
            print(f"  Error exporting {run_name}: {e}")
            return None
    
    def create_comparison_summary(self, all_results):
        """Create a comparison summary of all validation runs."""
        print("\nCreating comparison summary...")
        
        summary_data = []
        
        for run_name, data in all_results.items():
            if data is None:
                continue
            
            if isinstance(data, dict):  # Random split summary
                summary_data.append({
                    'run_name': run_name,
                    'validation_type': data.get('validation_type', 'unknown'),
                    'total_sites': 'N/A',
                    'mean_r2': data.get('overall_r2', 0),
                    'mean_rmse': data.get('overall_rmse', 0),
                    'median_r2': 'N/A',
                    'excellent_sites': 'N/A',
                    'good_sites': 'N/A',
                    'fair_sites': 'N/A',
                    'poor_sites': 'N/A',
                    'very_poor_sites': 'N/A'
                })
            else:  # Site-specific results
                summary_data.append({
                    'run_name': run_name,
                    'validation_type': data.attrs.get('validation_type', 'unknown'),
                    'total_sites': len(data),
                    'mean_r2': data['test_r2'].mean(),
                    'mean_rmse': data['test_rmse'].mean(),
                    'median_r2': data['test_r2'].median(),
                    'excellent_sites': len(data[data['category'] == 'Excellent']),
                    'good_sites': len(data[data['category'] == 'Good']),
                    'fair_sites': len(data[data['category'] == 'Fair']),
                    'poor_sites': len(data[data['category'] == 'Poor']),
                    'very_poor_sites': len(data[data['category'] == 'Very Poor'])
                })
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_data)
        
        # Export summary
        summary_path = self.output_dir / "validation_comparison_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"✓ Comparison summary exported: {summary_path}")
        
        # Print summary
        print("\nValidation Run Comparison:")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def create_visualization_guide(self, all_results):
        """Create a guide for QGIS visualization."""
        guide_path = self.output_dir / "qgis_visualization_guide.md"
        
        with open(guide_path, 'w') as f:
            f.write("# QGIS Visualization Guide for Validation Results\n\n")
            f.write("## Available GeoJSON Files:\n\n")
            
            for run_name, data in all_results.items():
                if data is not None and not isinstance(data, dict):
                    f.write(f"### {run_name}\n")
                    f.write(f"- **Validation Type:** {data.attrs.get('validation_type', 'unknown')}\n")
                    f.write(f"- **Total Sites:** {len(data)}\n")
                    f.write(f"- **Mean R²:** {data['test_r2'].mean():.3f}\n")
                    f.write(f"- **Mean RMSE:** {data['test_rmse'].mean():.3f}\n\n")
                    
                    f.write("**Styling Recommendations:**\n")
                    f.write("- Use 'test_r2' field for graduated symbology\n")
                    f.write("- Color ramp: RdYlGn (inverted)\n")
                    f.write("- Classes: 7 (Natural Breaks)\n")
                    f.write("- Symbol size: 3-5 mm\n\n")
            
            f.write("## Comparison Analysis:\n\n")
            f.write("1. **Load multiple GeoJSON files** as separate layers\n")
            f.write("2. **Use different colors** for each validation type\n")
            f.write("3. **Compare performance patterns** across validation strategies\n")
            f.write("4. **Identify consistent problem areas** across all methods\n")
            f.write("5. **Export comparison maps** using print layouts\n\n")
        
        print(f"✓ Visualization guide exported: {guide_path}")
    
    def run_pipeline(self):
        """Run the complete geographic performance pipeline."""
        print("Geographic Performance Pipeline")
        print("=" * 40)
        
        # Load spatial data
        if not self.load_spatial_data():
            return
        
        # Discover validation runs
        validation_runs = self.discover_validation_runs()
        if not validation_runs:
            print("No validation runs found!")
            return
        
        # Process each validation run
        all_results = {}
        
        for run_info in validation_runs:
            if run_info.get('is_random_split', False):
                result = self.process_random_split_results(run_info)
            else:
                result = self.process_site_specific_results(run_info)
            
            if result is not None:
                # Export GeoJSON
                if not isinstance(result, dict):  # Not a random split summary
                    self.export_geojson(result, run_info['name'])
                
                all_results[run_info['name']] = result
        
        # Create comparison summary
        if all_results:
            self.create_comparison_summary(all_results)
            self.create_visualization_guide(all_results)
        
        print("\n" + "=" * 50)
        print("PIPELINE COMPLETE")
        print("=" * 50)
        print(f"\nResults saved to: {self.output_dir}")
        print("\nNext steps:")
        print("1. Open QGIS")
        print("2. Load the generated GeoJSON files")
        print("3. Compare performance across validation strategies")
        print("4. Use the visualization guide for styling recommendations")

def main():
    """Main function to run the pipeline."""
    project_path = Path.cwd()
    pipeline = GeographicPerformancePipeline(project_path)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main() 