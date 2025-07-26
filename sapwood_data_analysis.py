#!/usr/bin/env python3
"""
SAPFLUXNET Data Analysis Script
===============================

Comprehensive analysis of COL_MAC_SAF_RAD dataset from Colombian Theobroma cacao site.
Uses memory-efficient processing with chunked data loading and streaming analysis.

Author: AI Assistant
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Optional
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class SapfluxAnalyzer:
    """Memory-efficient analyzer for SAPFLUXNET data."""
    
    def __init__(self, data_dir: str = "sapwood", output_dir: str = "sapwood_analysis_results"):
        """Initialize analyzer with data and output directories."""
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # File patterns for the dataset
        self.site_code = "COL_MAC_SAF_RAD"
        self.chunk_size = 100  # Process data in chunks to manage memory
        
        # Analysis results storage
        self.results = {
            'metadata': {},
            'data_quality': {},
            'temporal_analysis': {},
            'environmental_analysis': {},
            'sap_flow_analysis': {},
            'correlations': {}
        }
        
        print(f"🌳 SAPFLUXNET Analyzer initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Output directory: {self.output_dir}")
    
    def load_metadata(self) -> Dict:
        """Load all metadata files efficiently."""
        print("\n📋 Loading metadata files...")
        
        metadata_files = {
            'site': f"{self.site_code}_site_md.csv",
            'species': f"{self.site_code}_species_md.csv", 
            'plant': f"{self.site_code}_plant_md.csv",
            'stand': f"{self.site_code}_stand_md.csv",
            'env': f"{self.site_code}_env_md.csv"
        }
        
        metadata = {}
        for key, filename in metadata_files.items():
            filepath = self.data_dir / filename
            if filepath.exists():
                df = pd.read_csv(filepath)
                metadata[key] = df
                print(f"  ✓ Loaded {key} metadata: {len(df)} records")
            else:
                print(f"  ⚠️  Missing {key} metadata file: {filename}")
        
        self.results['metadata'] = metadata
        return metadata
    
    def analyze_data_quality(self) -> Dict:
        """Analyze data quality using flags and chunked processing."""
        print("\n🔍 Analyzing data quality...")
        
        quality_results = {
            'env_flags': {},
            'sapf_flags': {},
            'missing_data': {},
            'flag_distributions': {}
        }
        
        # Analyze environmental flags
        env_flags_file = self.data_dir / f"{self.site_code}_env_flags.csv"
        if env_flags_file.exists():
            print("  📊 Processing environmental flags...")
            flag_stats = {}
            
            # Process in chunks to manage memory
            for chunk in pd.read_csv(env_flags_file, chunksize=self.chunk_size):
                for col in chunk.columns:
                    if col not in ['TIMESTAMP', 'TIMESTAMP_solar']:
                        if col not in flag_stats:
                            flag_stats[col] = {}
                        
                        # Count flag types in this chunk
                        chunk_flags = chunk[col].dropna().value_counts()
                        for flag, count in chunk_flags.items():
                            flag_stats[col][flag] = flag_stats[col].get(flag, 0) + count
            
            quality_results['env_flags'] = flag_stats
        
        # Analyze sap flow flags
        sapf_flags_file = self.data_dir / f"{self.site_code}_sapf_flags.csv"
        if sapf_flags_file.exists():
            print("  🌲 Processing sap flow flags...")
            flag_stats = {}
            
            for chunk in pd.read_csv(sapf_flags_file, chunksize=self.chunk_size):
                for col in chunk.columns:
                    if col not in ['TIMESTAMP', 'TIMESTAMP_solar']:
                        if col not in flag_stats:
                            flag_stats[col] = {}
                        
                        chunk_flags = chunk[col].dropna().value_counts()
                        for flag, count in chunk_flags.items():
                            flag_stats[col][flag] = flag_stats[col].get(flag, 0) + count
            
            quality_results['sapf_flags'] = flag_stats
        
        self.results['data_quality'] = quality_results
        return quality_results
    
    def analyze_temporal_patterns(self) -> Dict:
        """Analyze temporal patterns in the data using efficient processing."""
        print("\n⏰ Analyzing temporal patterns...")
        
        temporal_results = {
            'time_range': {},
            'sampling_frequency': {},
            'daily_patterns': {},
            'seasonal_patterns': {}
        }
        
        # Load environmental data in chunks for temporal analysis
        env_file = self.data_dir / f"{self.site_code}_env_data.csv"
        sapf_file = self.data_dir / f"{self.site_code}_sapf_data.csv"
        
        if env_file.exists():
            # Get time range and frequency
            first_chunk = pd.read_csv(env_file, nrows=10)
            last_chunk = pd.read_csv(env_file, skiprows=range(1, pd.read_csv(env_file, nrows=0).shape[0]), nrows=10)
            
            first_chunk['TIMESTAMP'] = pd.to_datetime(first_chunk['TIMESTAMP'])
            last_chunk['TIMESTAMP'] = pd.to_datetime(last_chunk['TIMESTAMP'])
            
            temporal_results['time_range'] = {
                'start': first_chunk['TIMESTAMP'].iloc[0].isoformat(),
                'end': last_chunk['TIMESTAMP'].iloc[-1].isoformat(),
                'duration_days': (last_chunk['TIMESTAMP'].iloc[-1] - first_chunk['TIMESTAMP'].iloc[0]).days
            }
            
            # Analyze sampling frequency
            time_diff = (first_chunk['TIMESTAMP'].iloc[1] - first_chunk['TIMESTAMP'].iloc[0]).total_seconds() / 60
            temporal_results['sampling_frequency'] = f"{time_diff:.0f} minutes"
        
        self.results['temporal_analysis'] = temporal_results
        return temporal_results
    
    def analyze_environmental_data(self) -> Dict:
        """Analyze environmental variables using streaming approach."""
        print("\n🌤️  Analyzing environmental data...")
        
        env_file = self.data_dir / f"{self.site_code}_env_data.csv"
        if not env_file.exists():
            return {}
        
        env_results = {
            'statistics': {},
            'correlations': {},
            'daily_cycles': {}
        }
        
        # Process environmental data in chunks
        print("  📊 Computing environmental statistics...")
        env_stats = {}
        chunk_count = 0
        
        for chunk in pd.read_csv(env_file, chunksize=self.chunk_size):
            chunk['TIMESTAMP'] = pd.to_datetime(chunk['TIMESTAMP'])
            
            # Calculate statistics for numeric columns
            numeric_cols = chunk.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                if col not in env_stats:
                    env_stats[col] = {'sum': 0, 'sum_sq': 0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
                
                values = chunk[col].dropna()
                if len(values) > 0:
                    env_stats[col]['sum'] += values.sum()
                    env_stats[col]['sum_sq'] += (values ** 2).sum()
                    env_stats[col]['count'] += len(values)
                    env_stats[col]['min'] = min(env_stats[col]['min'], values.min())
                    env_stats[col]['max'] = max(env_stats[col]['max'], values.max())
            
            chunk_count += 1
        
        # Calculate final statistics
        final_stats = {}
        for col, stats in env_stats.items():
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(0, variance))
                
                final_stats[col] = {
                    'mean': round(mean, 4),
                    'std': round(std, 4),
                    'min': round(stats['min'], 4),
                    'max': round(stats['max'], 4),
                    'count': stats['count']
                }
        
        env_results['statistics'] = final_stats
        self.results['environmental_analysis'] = env_results
        return env_results
    
    def analyze_sap_flow(self) -> Dict:
        """Analyze sap flow data for the three trees."""
        print("\n🌲 Analyzing sap flow data...")
        
        sapf_file = self.data_dir / f"{self.site_code}_sapf_data.csv"
        if not sapf_file.exists():
            return {}
        
        sapf_results = {
            'tree_statistics': {},
            'treatment_comparison': {},
            'daily_patterns': {}
        }
        
        # Get plant metadata for treatment information
        plant_md = self.results['metadata'].get('plant', pd.DataFrame())
        
        # Process sap flow data in chunks
        print("  📊 Computing sap flow statistics...")
        sapf_stats = {}
        chunk_count = 0
        
        for chunk in pd.read_csv(sapf_file, chunksize=self.chunk_size):
            chunk['TIMESTAMP'] = pd.to_datetime(chunk['TIMESTAMP'])
            
            # Get sap flow columns (exclude timestamp columns)
            sapf_cols = [col for col in chunk.columns if col.startswith('COL_MAC_SAF_Tca_Js_')]
            
            for col in sapf_cols:
                if col not in sapf_stats:
                    sapf_stats[col] = {'sum': 0, 'sum_sq': 0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
                
                values = chunk[col].dropna()
                if len(values) > 0:
                    sapf_stats[col]['sum'] += values.sum()
                    sapf_stats[col]['sum_sq'] += (values ** 2).sum()
                    sapf_stats[col]['count'] += len(values)
                    sapf_stats[col]['min'] = min(sapf_stats[col]['min'], values.min())
                    sapf_stats[col]['max'] = max(sapf_stats[col]['max'], values.max())
            
            chunk_count += 1
        
        # Calculate final statistics and add treatment info
        tree_stats = {}
        for col, stats in sapf_stats.items():
            if stats['count'] > 0:
                mean = stats['sum'] / stats['count']
                variance = (stats['sum_sq'] / stats['count']) - (mean ** 2)
                std = np.sqrt(max(0, variance))
                
                # Get treatment info from plant metadata
                treatment = "Unknown"
                if not plant_md.empty:
                    plant_row = plant_md[plant_md['pl_code'] == col]
                    if not plant_row.empty:
                        treatment = plant_row['pl_treatment'].iloc[0]
                
                tree_stats[col] = {
                    'mean': round(mean, 6),
                    'std': round(std, 6),
                    'min': round(stats['min'], 6),
                    'max': round(stats['max'], 6),
                    'count': stats['count'],
                    'treatment': treatment
                }
        
        sapf_results['tree_statistics'] = tree_stats
        self.results['sap_flow_analysis'] = sapf_results
        return sapf_results
    
    def create_visualizations(self) -> None:
        """Create visualizations using efficient data loading."""
        print("\n📊 Creating visualizations...")
        
        # Set up matplotlib for better plots
        plt.style.use('default')
        fig_size = (12, 8)
        
        # Create summary plots
        self._plot_data_quality_summary()
        self._plot_environmental_summary()
        self._plot_sap_flow_comparison()
        
        print(f"  ✓ Visualizations saved to {self.output_dir}")
    
    def _plot_data_quality_summary(self) -> None:
        """Plot data quality summary."""
        if 'data_quality' not in self.results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Quality Summary - COL_MAC_SAF_RAD', fontsize=16, fontweight='bold')
        
        # Environmental flags
        env_flags = self.results['data_quality'].get('env_flags', {})
        if env_flags:
            ax = axes[0, 0]
            flag_counts = {}
            for var, flags in env_flags.items():
                for flag, count in flags.items():
                    flag_counts[flag] = flag_counts.get(flag, 0) + count
            
            if flag_counts:
                ax.bar(flag_counts.keys(), flag_counts.values())
                ax.set_title('Environmental Data Flags')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
        
        # Sap flow flags
        sapf_flags = self.results['data_quality'].get('sapf_flags', {})
        if sapf_flags:
            ax = axes[0, 1]
            flag_counts = {}
            for var, flags in sapf_flags.items():
                for flag, count in flags.items():
                    flag_counts[flag] = flag_counts.get(flag, 0) + count
            
            if flag_counts:
                ax.bar(flag_counts.keys(), flag_counts.values())
                ax.set_title('Sap Flow Data Flags')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_quality_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_environmental_summary(self) -> None:
        """Plot environmental data summary."""
        env_stats = self.results.get('environmental_analysis', {}).get('statistics', {})
        if not env_stats:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Environmental Variables Summary - COL_MAC_SAF_RAD', fontsize=16, fontweight='bold')
        
        variables = ['ta', 'rh', 'vpd', 'sw_in']
        var_labels = ['Air Temperature (°C)', 'Relative Humidity (%)', 'VPD (kPa)', 'Solar Radiation (W/m²)']
        
        for i, (var, label) in enumerate(zip(variables, var_labels)):
            if var in env_stats:
                ax = axes[i//2, i%2]
                stats = env_stats[var]
                
                # Create a simple bar plot of statistics
                ax.bar(['Mean', 'Min', 'Max'], [stats['mean'], stats['min'], stats['max']])
                ax.set_title(label)
                ax.set_ylabel('Value')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'environmental_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_sap_flow_comparison(self) -> None:
        """Plot sap flow comparison between trees."""
        tree_stats = self.results.get('sap_flow_analysis', {}).get('tree_statistics', {})
        if not tree_stats:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Sap Flow Comparison Between Trees - COL_MAC_SAF_RAD', fontsize=16, fontweight='bold')
        
        trees = list(tree_stats.keys())
        means = [tree_stats[tree]['mean'] for tree in trees]
        treatments = [tree_stats[tree]['treatment'] for tree in trees]
        
        # Mean sap flow by tree
        bars = ax1.bar(range(len(trees)), means)
        ax1.set_xlabel('Tree')
        ax1.set_ylabel('Mean Sap Flow (cm³ cm⁻² h⁻¹)')
        ax1.set_title('Mean Sap Flow by Tree')
        ax1.set_xticks(range(len(trees)))
        ax1.set_xticklabels([f"Tree {i+1}" for i in range(len(trees))])
        
        # Add treatment labels
        for i, (bar, treatment) in enumerate(zip(bars, treatments)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    treatment, ha='center', va='bottom', rotation=45, fontsize=8)
        
        # Standard deviation comparison
        stds = [tree_stats[tree]['std'] for tree in trees]
        ax2.bar(range(len(trees)), stds)
        ax2.set_xlabel('Tree')
        ax2.set_ylabel('Std Dev Sap Flow (cm³ cm⁻² h⁻¹)')
        ax2.set_title('Sap Flow Variability by Tree')
        ax2.set_xticks(range(len(trees)))
        ax2.set_xticklabels([f"Tree {i+1}" for i in range(len(trees))])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sap_flow_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self) -> None:
        """Generate comprehensive analysis report."""
        print("\n📝 Generating comprehensive report...")
        
        report_file = self.output_dir / f"sapflux_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("SAPFLUXNET DATA ANALYSIS REPORT\n")
            f.write("="*80 + "\n")
            f.write(f"Site: COL_MAC_SAF_RAD (Macagual Universidad de la Amazonia)\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Species: Theobroma cacao (Cocoa)\n")
            f.write("="*80 + "\n\n")
            
            # Site Information
            if 'metadata' in self.results and 'site' in self.results['metadata']:
                site_md = self.results['metadata']['site']
                if not site_md.empty:
                    f.write("SITE INFORMATION\n")
                    f.write("-" * 40 + "\n")
                    f.write(f"Location: {site_md['si_country'].iloc[0]}\n")
                    f.write(f"Coordinates: {site_md['si_lat'].iloc[0]:.6f}°N, {site_md['si_long'].iloc[0]:.6f}°W\n")
                    f.write(f"Elevation: {site_md['si_elev'].iloc[0]} m\n")
                    f.write(f"Biome: {site_md['si_biome'].iloc[0]}\n")
                    f.write(f"Climate: MAT = {site_md['si_mat'].iloc[0]:.1f}°C, MAP = {site_md['si_map'].iloc[0]:.0f} mm\n\n")
            
            # Species and Stand Information
            if 'metadata' in self.results:
                if 'species' in self.results['metadata']:
                    species_md = self.results['metadata']['species']
                    if not species_md.empty:
                        f.write("SPECIES INFORMATION\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Species: {species_md['sp_name'].iloc[0]}\n")
                        f.write(f"Leaf Habit: {species_md['sp_leaf_habit'].iloc[0]}\n")
                        f.write(f"Number of Trees: {species_md['sp_ntrees'].iloc[0]}\n\n")
                
                if 'stand' in self.results['metadata']:
                    stand_md = self.results['metadata']['stand']
                    if not stand_md.empty:
                        f.write("STAND CHARACTERISTICS\n")
                        f.write("-" * 40 + "\n")
                        f.write(f"Stand Age: {stand_md['st_age'].iloc[0]} years\n")
                        f.write(f"Stand Height: {stand_md['st_height'].iloc[0]} m\n")
                        f.write(f"LAI: {stand_md['st_lai'].iloc[0]}\n")
                        f.write(f"Basal Area: {stand_md['st_basal_area'].iloc[0]} m²/ha\n")
                        f.write(f"Tree Density: {stand_md['st_density'].iloc[0]} trees/ha\n")
                        f.write(f"Soil Texture: {stand_md['st_USDA_soil_texture'].iloc[0]}\n\n")
            
            # Temporal Analysis
            if 'temporal_analysis' in self.results:
                temporal = self.results['temporal_analysis']
                f.write("TEMPORAL COVERAGE\n")
                f.write("-" * 40 + "\n")
                if 'time_range' in temporal:
                    tr = temporal['time_range']
                    f.write(f"Start Date: {tr.get('start', 'Unknown')}\n")
                    f.write(f"End Date: {tr.get('end', 'Unknown')}\n")
                    f.write(f"Duration: {tr.get('duration_days', 'Unknown')} days\n")
                if 'sampling_frequency' in temporal:
                    f.write(f"Sampling Frequency: {temporal['sampling_frequency']}\n\n")
            
            # Environmental Data Analysis
            if 'environmental_analysis' in self.results:
                env_analysis = self.results['environmental_analysis']
                f.write("ENVIRONMENTAL CONDITIONS\n")
                f.write("-" * 40 + "\n")
                
                if 'statistics' in env_analysis:
                    stats = env_analysis['statistics']
                    var_names = {
                        'ta': 'Air Temperature (°C)',
                        'rh': 'Relative Humidity (%)',
                        'vpd': 'Vapor Pressure Deficit (kPa)',
                        'ppfd_in': 'PPFD (μmol m⁻² s⁻¹)',
                        'sw_in': 'Solar Radiation (W m⁻²)'
                    }
                    
                    for var, name in var_names.items():
                        if var in stats:
                            s = stats[var]
                            f.write(f"{name}:\n")
                            f.write(f"  Mean: {s['mean']:.2f}, Std: {s['std']:.2f}\n")
                            f.write(f"  Range: {s['min']:.2f} to {s['max']:.2f}\n")
                            f.write(f"  Valid observations: {s['count']}\n\n")
            
            # Sap Flow Analysis
            if 'sap_flow_analysis' in self.results:
                sapf_analysis = self.results['sap_flow_analysis']
                f.write("SAP FLOW ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                if 'tree_statistics' in sapf_analysis:
                    tree_stats = sapf_analysis['tree_statistics']
                    
                    for i, (tree_id, stats) in enumerate(tree_stats.items(), 1):
                        f.write(f"Tree {i} ({tree_id}):\n")
                        f.write(f"  Treatment: {stats['treatment']}\n")
                        f.write(f"  Mean Sap Flow: {stats['mean']:.6f} cm³ cm⁻² h⁻¹\n")
                        f.write(f"  Std Dev: {stats['std']:.6f} cm³ cm⁻² h⁻¹\n")
                        f.write(f"  Range: {stats['min']:.6f} to {stats['max']:.6f} cm³ cm⁻² h⁻¹\n")
                        f.write(f"  Valid observations: {stats['count']}\n\n")
            
            # Data Quality Summary
            if 'data_quality' in self.results:
                data_quality = self.results['data_quality']
                f.write("DATA QUALITY ASSESSMENT\n")
                f.write("-" * 40 + "\n")
                
                if 'env_flags' in data_quality:
                    f.write("Environmental Data Flags:\n")
                    for var, flags in data_quality['env_flags'].items():
                        if flags:
                            f.write(f"  {var}: {', '.join([f'{flag}({count})' for flag, count in flags.items()])}\n")
                    f.write("\n")
                
                if 'sapf_flags' in data_quality:
                    f.write("Sap Flow Data Flags:\n")
                    for var, flags in data_quality['sapf_flags'].items():
                        if flags:
                            f.write(f"  {var}: {', '.join([f'{flag}({count})' for flag, count in flags.items()])}\n")
                    f.write("\n")
            
            f.write("="*80 + "\n")
            f.write("Analysis completed using memory-efficient chunked processing\n")
            f.write("Visualizations and detailed results available in output directory\n")
            f.write("="*80 + "\n")
        
        print(f"  ✓ Comprehensive report saved: {report_file}")
    
    def save_results(self) -> None:
        """Save analysis results to JSON file."""
        print("\n💾 Saving analysis results...")
        
        results_file = self.output_dir / f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert non-serializable objects for JSON
        serializable_results = {}
        for key, value in self.results.items():
            if key == 'metadata':
                # Convert DataFrames to dictionaries
                serializable_results[key] = {k: v.to_dict('records') if hasattr(v, 'to_dict') else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = value
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        print(f"  ✓ Results saved: {results_file}")
    
    def run_full_analysis(self) -> None:
        """Run the complete analysis pipeline."""
        print("🚀 Starting comprehensive SAPFLUXNET analysis...")
        print("=" * 60)
        
        try:
            # Load metadata
            self.load_metadata()
            
            # Analyze data quality
            self.analyze_data_quality()
            
            # Temporal analysis
            self.analyze_temporal_patterns()
            
            # Environmental analysis
            self.analyze_environmental_data()
            
            # Sap flow analysis
            self.analyze_sap_flow()
            
            # Create visualizations
            self.create_visualizations()
            
            # Generate report
            self.generate_report()
            
            # Save results
            self.save_results()
            
            print("\n" + "=" * 60)
            print("✅ Analysis completed successfully!")
            print(f"📁 Results available in: {self.output_dir}")
            print("📊 Generated files:")
            print("  - Comprehensive analysis report (.txt)")
            print("  - Data quality visualizations (.png)")
            print("  - Environmental summary plots (.png)")
            print("  - Sap flow comparison charts (.png)")
            print("  - Complete results dataset (.json)")
            
        except Exception as e:
            print(f"\n❌ Error during analysis: {str(e)}")
            raise


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Comprehensive SAPFLUXNET data analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        default="sapwood",
        help="Directory containing SAPFLUXNET CSV files"
    )
    
    parser.add_argument(
        "--output-dir", 
        default="sapwood_analysis_results",
        help="Directory for analysis outputs"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100,
        help="Chunk size for memory-efficient processing"
    )
    
    args = parser.parse_args()
    
    # Initialize and run analyzer
    analyzer = SapfluxAnalyzer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    analyzer.chunk_size = args.chunk_size
    
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main() 