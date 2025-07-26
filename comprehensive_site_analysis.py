#!/usr/bin/env python3
"""
Comprehensive SAPFLUXNET Site Analysis
=====================================

Analyzes each site across multiple dimensions to understand:
- Temporal coverage and representativeness  
- Site characteristics and metadata
- Environmental variable ranges and distributions
- Geographic and climatic outliers
- Potential issues for spatial modeling
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

class ComprehensiveSiteAnalyzer:
    def __init__(self, data_dir="sapwood"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("comprehensive_site_analysis")
        self.output_dir.mkdir(exist_ok=True)
        
        # Define what makes a site representative for modeling
        self.representativeness_criteria = {
            'min_duration_days': 30,  # At least 30 days of data
            'min_observations': 500,  # At least 500 observations
            'min_environmental_coverage': 0.8,  # 80% of critical variables present
            'max_missing_rate': 0.3,  # No more than 30% missing data
            'seasonal_coverage_threshold': 0.25  # At least 1/4 of a year
        }
        
        # Critical environmental variables for ecological modeling
        self.critical_env_vars = ['ta', 'rh', 'vpd', 'ppfd_in', 'sw_in']
        
        print(f"🔬 Comprehensive Site Analyzer initialized")
        print(f"📁 Data directory: {self.data_dir}")
        print(f"📊 Output directory: {self.output_dir}")
    
    def discover_sites(self):
        """Find all sites with complete datasets."""
        print("\n🌍 Discovering SAPFLUXNET sites...")
        
        env_files = list(self.data_dir.glob("*_env_data.csv"))
        sites_info = {}
        
        for env_file in env_files:
            site_code = env_file.stem.replace('_env_data', '')
            
            # Check for associated files
            base_path = self.data_dir / site_code
            files_present = {
                'env_data': env_file.exists(),
                'sapf_data': (self.data_dir / f"{site_code}_sapf_data.csv").exists(),
                'site_md': (self.data_dir / f"{site_code}_site_md.csv").exists(),
                'species_md': (self.data_dir / f"{site_code}_species_md.csv").exists(),
                'plant_md': (self.data_dir / f"{site_code}_plant_md.csv").exists(),
                'stand_md': (self.data_dir / f"{site_code}_stand_md.csv").exists()
            }
            
            sites_info[site_code] = {
                'env_file': env_file,
                'files_present': files_present,
                'completeness': sum(files_present.values()) / len(files_present)
            }
        
        print(f"  ✓ Found {len(sites_info)} sites")
        return sites_info
    
    def load_metadata(self, site_code):
        """Load all available metadata for a site."""
        metadata = {}
        
        metadata_files = {
            'site': f"{site_code}_site_md.csv",
            'species': f"{site_code}_species_md.csv", 
            'plant': f"{site_code}_plant_md.csv",
            'stand': f"{site_code}_stand_md.csv",
            'env': f"{site_code}_env_md.csv"
        }
        
        for md_type, filename in metadata_files.items():
            file_path = self.data_dir / filename
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if len(df) > 0:
                        # Convert to dict, taking first row
                        metadata[md_type] = df.iloc[0].to_dict()
                except Exception as e:
                    metadata[md_type] = {'error': str(e)}
            else:
                metadata[md_type] = None
        
        return metadata
    
    def analyze_temporal_coverage(self, site_code, env_file):
        """Analyze temporal characteristics of the site."""
        print(f"    📅 Analyzing temporal coverage...")
        
        try:
            # Read timestamps (first few chunks to get coverage)
            df_sample = pd.read_csv(env_file, nrows=10000)
            
            if 'TIMESTAMP' not in df_sample.columns:
                return {'error': 'No TIMESTAMP column found'}
            
            # Parse timestamps
            timestamps = pd.to_datetime(df_sample['TIMESTAMP'], errors='coerce')
            valid_timestamps = timestamps.dropna()
            
            if len(valid_timestamps) < 2:
                return {'error': 'Insufficient valid timestamps'}
            
            # Calculate temporal characteristics
            start_date = valid_timestamps.min()
            end_date = valid_timestamps.max()
            duration = end_date - start_date
            total_records = len(df_sample)
            
            # Estimate sampling frequency
            time_diffs = valid_timestamps.diff().dropna()
            if len(time_diffs) > 0:
                median_interval = time_diffs.median()
                sampling_freq_minutes = median_interval.total_seconds() / 60
            else:
                sampling_freq_minutes = None
            
            # Calculate representativeness scores
            duration_days = duration.total_seconds() / (24 * 3600)
            duration_score = min(100, (duration_days / self.representativeness_criteria['min_duration_days']) * 100)
            
            records_score = min(100, (total_records / self.representativeness_criteria['min_observations']) * 100)
            
            # Seasonal coverage (rough estimate)
            seasonal_coverage = min(1.0, duration_days / 365.25)
            seasonal_score = seasonal_coverage * 100
            
            return {
                'start_date': start_date.isoformat() if pd.notna(start_date) else None,
                'end_date': end_date.isoformat() if pd.notna(end_date) else None,
                'duration_days': round(duration_days, 2),
                'total_records': total_records,
                'sampling_freq_minutes': round(sampling_freq_minutes, 1) if sampling_freq_minutes else None,
                'duration_score': round(duration_score, 1),
                'records_score': round(records_score, 1),
                'seasonal_coverage': round(seasonal_coverage, 3),
                'seasonal_score': round(seasonal_score, 1),
                'temporal_representativeness': round((duration_score + records_score + seasonal_score) / 3, 1)
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_environmental_characteristics(self, site_code, env_file):
        """Analyze environmental data characteristics."""
        print(f"    🌡️ Analyzing environmental data...")
        
        try:
            # Read environmental data in chunks for memory efficiency
            env_stats = {}
            chunk_size = 1000
            
            for chunk in pd.read_csv(env_file, chunksize=chunk_size):
                for col in chunk.columns:
                    if col in ['TIMESTAMP', 'TIMESTAMP_solar', 'solar_TIMESTAMP']:
                        continue
                    
                    # Convert to numeric
                    numeric_vals = pd.to_numeric(chunk[col], errors='coerce')
                    valid_vals = numeric_vals.dropna()
                    
                    if col not in env_stats:
                        env_stats[col] = {
                            'total_count': 0,
                            'valid_count': 0,
                            'sum_vals': 0.0,
                            'sum_squared': 0.0,
                            'min_val': float('inf'),
                            'max_val': float('-inf'),
                            'zero_count': 0
                        }
                    
                    stats = env_stats[col]
                    stats['total_count'] += len(chunk[col])
                    stats['valid_count'] += len(valid_vals)
                    
                    if len(valid_vals) > 0:
                        stats['sum_vals'] += valid_vals.sum()
                        stats['sum_squared'] += (valid_vals ** 2).sum()
                        stats['min_val'] = min(stats['min_val'], valid_vals.min())
                        stats['max_val'] = max(stats['max_val'], valid_vals.max())
                        stats['zero_count'] += (valid_vals == 0).sum()
            
            # Calculate final statistics
            environmental_summary = {}
            critical_vars_present = 0
            total_missing_rate = 0
            
            for var, stats in env_stats.items():
                if stats['total_count'] > 0:
                    missing_rate = 1 - (stats['valid_count'] / stats['total_count'])
                    zero_rate = stats['zero_count'] / stats['total_count'] if stats['total_count'] > 0 else 0
                    
                    if stats['valid_count'] > 0:
                        mean_val = stats['sum_vals'] / stats['valid_count']
                        variance = (stats['sum_squared'] / stats['valid_count']) - (mean_val ** 2)
                        std_val = np.sqrt(max(0, variance))
                    else:
                        mean_val = std_val = 0
                    
                    environmental_summary[var] = {
                        'mean': round(mean_val, 3),
                        'std': round(std_val, 3),
                        'min': round(stats['min_val'], 3) if stats['min_val'] != float('inf') else None,
                        'max': round(stats['max_val'], 3) if stats['max_val'] != float('-inf') else None,
                        'missing_rate': round(missing_rate, 3),
                        'zero_rate': round(zero_rate, 3),
                        'valid_observations': stats['valid_count']
                    }
                    
                    total_missing_rate += missing_rate
                    
                    if var in self.critical_env_vars:
                        critical_vars_present += 1
            
            # Calculate environmental representativeness
            critical_coverage = critical_vars_present / len(self.critical_env_vars)
            avg_missing_rate = total_missing_rate / len(env_stats) if env_stats else 1.0
            data_completeness = 1 - avg_missing_rate
            
            env_representativeness = (critical_coverage * 50) + (data_completeness * 50)
            
            return {
                'variables_found': list(environmental_summary.keys()),
                'critical_vars_present': critical_vars_present,
                'critical_vars_total': len(self.critical_env_vars),
                'critical_coverage': round(critical_coverage, 3),
                'avg_missing_rate': round(avg_missing_rate, 3),
                'data_completeness': round(data_completeness, 3),
                'environmental_representativeness': round(env_representativeness, 1),
                'variable_stats': environmental_summary
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_site_outlier_score(self, site_analysis, all_sites_summary):
        """Calculate how much of an outlier this site is compared to others."""
        # This will be implemented after we have analyzed several sites
        # For now, return a placeholder
        return {
            'geographic_outlier_score': 0,
            'temporal_outlier_score': 0,
            'environmental_outlier_score': 0,
            'overall_outlier_score': 0
        }
    
    def analyze_single_site(self, site_code, site_info):
        """Perform comprehensive analysis of a single site."""
        print(f"\n🔬 Analyzing {site_code}...")
        
        analysis = {
            'site_code': site_code,
            'analysis_timestamp': datetime.now().isoformat(),
            'file_completeness': site_info['completeness']
        }
        
        # Load metadata
        print(f"    📋 Loading metadata...")
        analysis['metadata'] = self.load_metadata(site_code)
        
        # Temporal analysis
        analysis['temporal'] = self.analyze_temporal_coverage(site_code, site_info['env_file'])
        
        # Environmental analysis
        analysis['environmental'] = self.analyze_environmental_characteristics(site_code, site_info['env_file'])
        
        # Calculate overall representativeness for modeling
        temporal_score = analysis['temporal'].get('temporal_representativeness', 0)
        environmental_score = analysis['environmental'].get('environmental_representativeness', 0)
        
        # Overall modeling suitability
        modeling_suitability = (temporal_score * 0.4) + (environmental_score * 0.4) + (site_info['completeness'] * 100 * 0.2)
        analysis['modeling_suitability_score'] = round(modeling_suitability, 1)
        
        # Classify site for modeling
        if modeling_suitability >= 80:
            analysis['modeling_classification'] = 'EXCELLENT'
        elif modeling_suitability >= 60:
            analysis['modeling_classification'] = 'GOOD'
        elif modeling_suitability >= 40:
            analysis['modeling_classification'] = 'MARGINAL'
        else:
            analysis['modeling_classification'] = 'POOR'
        
        return analysis
    
    def run_comprehensive_analysis(self):
        """Run comprehensive analysis on all sites."""
        print("🚀 Starting comprehensive SAPFLUXNET site analysis...")
        print("=" * 70)
        
        # Discover sites
        sites_info = self.discover_sites()
        
        all_analyses = {}
        modeling_categories = {'EXCELLENT': [], 'GOOD': [], 'MARGINAL': [], 'POOR': []}
        
        # Analyze each site
        for i, (site_code, site_info) in enumerate(sites_info.items(), 1):
            print(f"\n[{i:3d}/{len(sites_info)}] Processing {site_code}...")
            
            try:
                analysis = self.analyze_single_site(site_code, site_info)
                all_analyses[site_code] = analysis
                
                # Categorize for modeling
                category = analysis['modeling_classification']
                modeling_categories[category].append(site_code)
                
            except Exception as e:
                print(f"    ❌ Error analyzing {site_code}: {str(e)}")
                all_analyses[site_code] = {
                    'site_code': site_code,
                    'error': str(e),
                    'modeling_classification': 'ERROR'
                }
        
        # Generate comprehensive report
        self.generate_comprehensive_report(all_analyses, modeling_categories)
        
        # Save detailed results
        results_file = self.output_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_analyses, f, indent=2, default=str)
        
        print(f"\n✅ Comprehensive analysis complete!")
        print(f"📊 Analyzed {len(all_analyses)} sites")
        print(f"📝 Detailed results saved: {results_file}")
        
        # Summary by category
        for category, sites in modeling_categories.items():
            if sites:
                print(f"🏷️  {category}: {len(sites)} sites")
        
        return all_analyses
    
    def generate_comprehensive_report(self, all_analyses, modeling_categories):
        """Generate comprehensive report with insights for spatial modeling."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"comprehensive_site_report_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("COMPREHENSIVE SAPFLUXNET SITE ANALYSIS REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write("Analysis Focus: Site representativeness for spatial modeling\n")
            f.write("Identifies: Temporal coverage, data quality, and outlier characteristics\n\n")
            
            # Summary statistics
            successful_analyses = [a for a in all_analyses.values() if 'error' not in a]
            if successful_analyses:
                scores = [a['modeling_suitability_score'] for a in successful_analyses]
                f.write(f"Sites analyzed: {len(all_analyses)}\n")
                f.write(f"Successful analyses: {len(successful_analyses)}\n")
                f.write(f"Average modeling suitability: {np.mean(scores):.1f}/100\n")
                f.write(f"Range: {np.min(scores):.1f} to {np.max(scores):.1f}\n\n")
            
            # Modeling categories
            f.write("MODELING SUITABILITY CATEGORIES:\n")
            f.write("-" * 40 + "\n")
            for category, sites in modeling_categories.items():
                f.write(f"{category}: {len(sites)} sites\n")
                if sites and len(sites) <= 10:
                    f.write(f"  Sites: {', '.join(sites)}\n")
                elif sites:
                    f.write(f"  First 10: {', '.join(sites[:10])}\n")
            f.write("\n")
            
            # Sites with poor temporal coverage (like COL_MAC_SAF_RAD)
            poor_temporal = []
            for site_code, analysis in all_analyses.items():
                if ('temporal' in analysis and 
                    analysis['temporal'].get('duration_days', 0) < self.representativeness_criteria['min_duration_days']):
                    duration = analysis['temporal'].get('duration_days', 0)
                    poor_temporal.append((site_code, duration))
            
            if poor_temporal:
                f.write("SITES WITH POOR TEMPORAL COVERAGE (< 30 days):\n")
                f.write("-" * 45 + "\n")
                poor_temporal.sort(key=lambda x: x[1])
                for site, duration in poor_temporal:
                    f.write(f"{site}: {duration:.1f} days\n")
                f.write("\n")
            
            # Sites with missing critical environmental variables
            missing_env = []
            for site_code, analysis in all_analyses.items():
                if ('environmental' in analysis and 
                    analysis['environmental'].get('critical_coverage', 1) < 0.8):
                    coverage = analysis['environmental'].get('critical_coverage', 0)
                    missing_env.append((site_code, coverage))
            
            if missing_env:
                f.write("SITES WITH INCOMPLETE ENVIRONMENTAL DATA (< 80% critical variables):\n")
                f.write("-" * 65 + "\n")
                missing_env.sort(key=lambda x: x[1])
                for site, coverage in missing_env:
                    f.write(f"{site}: {coverage:.1%} coverage\n")
                f.write("\n")
            
            # Detailed analysis for worst performers
            worst_sites = []
            for site_code, analysis in all_analyses.items():
                if 'modeling_suitability_score' in analysis:
                    score = analysis['modeling_suitability_score']
                    if score < 50:  # Poor performers
                        worst_sites.append((site_code, score, analysis))
            
            if worst_sites:
                f.write("DETAILED ANALYSIS - WORST PERFORMING SITES:\n")
                f.write("-" * 45 + "\n")
                worst_sites.sort(key=lambda x: x[1])
                
                for i, (site_code, score, analysis) in enumerate(worst_sites[:10], 1):
                    f.write(f"\n{i:2d}. {site_code} (Score: {score:.1f}/100)\n")
                    
                    # Temporal issues
                    if 'temporal' in analysis:
                        temporal = analysis['temporal']
                        if temporal.get('duration_days', 0) < 30:
                            f.write(f"    ⚠️  Very short duration: {temporal.get('duration_days', 0):.1f} days\n")
                        if temporal.get('total_records', 0) < 500:
                            f.write(f"    ⚠️  Few observations: {temporal.get('total_records', 0)}\n")
                    
                    # Environmental issues
                    if 'environmental' in analysis:
                        env = analysis['environmental']
                        if env.get('critical_coverage', 1) < 0.8:
                            f.write(f"    ⚠️  Missing critical variables: {env.get('critical_coverage', 0):.1%} coverage\n")
                        if env.get('avg_missing_rate', 0) > 0.3:
                            f.write(f"    ⚠️  High missing data rate: {env.get('avg_missing_rate', 0):.1%}\n")
                    
                    # Metadata insights
                    if 'metadata' in analysis:
                        site_md = analysis['metadata'].get('site', {})
                        if site_md and isinstance(site_md, dict):
                            if 'si_elev' in site_md:
                                f.write(f"    📍 Elevation: {site_md['si_elev']} m\n")
                            if 'si_biome' in site_md:
                                f.write(f"    🌍 Biome: {site_md['si_biome']}\n")
        
        print(f"📝 Comprehensive report saved: {report_file}")

if __name__ == "__main__":
    analyzer = ComprehensiveSiteAnalyzer()
    analyzer.run_comprehensive_analysis() 