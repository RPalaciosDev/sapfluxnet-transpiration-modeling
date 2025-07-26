#!/usr/bin/env python3
"""
Ecological Data Quality Analyzer - Focused on modeling variables
Ignores timestamps, focuses on ecological data quality issues
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

class EcoQualityAnalyzer:
    def __init__(self, data_dir="sapwood"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path("eco_quality_results")
        self.output_dir.mkdir(exist_ok=True)
        
        # Variables to ignore (timestamps, technical)
        self.ignore_vars = {'TIMESTAMP', 'TIMESTAMP_solar', 'solar_TIMESTAMP', 'DATE', 'DOY'}
        
        # Critical ecological variables
        self.critical_eco_vars = {
            'ta': {'name': 'Air Temperature', 'zero_threshold': 2, 'weight': 2.0},
            'rh': {'name': 'Relative Humidity', 'zero_threshold': 1, 'weight': 2.0}, 
            'vpd': {'name': 'VPD', 'zero_threshold': 5, 'weight': 3.0},  # Most critical
            'ppfd_in': {'name': 'PPFD', 'zero_threshold': 45, 'weight': 1.0},
            'sw_in': {'name': 'Solar Radiation', 'zero_threshold': 45, 'weight': 1.0}
        }
    
    def analyze_site(self, site_code):
        env_file = self.data_dir / f"{site_code}_env_data.csv"
        if not env_file.exists():
            return None
        
        print(f"  🌱 {site_code}")
        
        # Read and analyze data
        try:
            df = pd.read_csv(env_file)
            
            # Remove ignored columns
            eco_cols = [col for col in df.columns if col not in self.ignore_vars]
            eco_data = df[eco_cols]
            
            results = {
                'site_code': site_code,
                'total_records': len(df),
                'issues': [],
                'zero_stats': {},
                'eco_score': 100.0
            }
            
            # Analyze each ecological variable
            for col in eco_cols:
                if col in self.critical_eco_vars:
                    # Convert to numeric, handle mixed types
                    numeric_vals = pd.to_numeric(eco_data[col], errors='coerce')
                    valid_vals = numeric_vals.dropna()
                    
                    if len(valid_vals) > 0:
                        zero_count = (valid_vals == 0).sum()
                        zero_pct = (zero_count / len(valid_vals)) * 100
                        threshold = self.critical_eco_vars[col]['zero_threshold']
                        
                        results['zero_stats'][col] = {
                            'zero_pct': round(zero_pct, 1),
                            'threshold': threshold,
                            'problematic': zero_pct > threshold
                        }
                        
                        # Score penalty for problematic variables
                        if zero_pct > threshold:
                            var_info = self.critical_eco_vars[col]
                            penalty = min(40, (zero_pct - threshold) * var_info['weight'] * 0.5)
                            results['eco_score'] -= penalty
                            
                            results['issues'].append(
                                f"{var_info['name']}: {zero_pct:.1f}% zeros (threshold: {threshold}%)"
                            )
            
            results['eco_score'] = max(0, round(results['eco_score'], 1))
            return results
            
        except Exception as e:
            return {'site_code': site_code, 'error': str(e)}
    
    def run_analysis(self):
        print("🌱 Ecological Data Quality Analysis (Timestamp-Ignored)")
        print("=" * 60)
        
        # Find all sites
        site_files = list(self.data_dir.glob("*_env_data.csv"))
        site_codes = [f.stem.replace('_env_data', '') for f in site_files]
        site_codes.sort()
        
        print(f"Found {len(site_codes)} sites")
        
        all_results = []
        problematic_sites = []
        
        for i, site_code in enumerate(site_codes, 1):
            print(f"[{i:3d}/{len(site_codes)}]", end=" ")
            result = self.analyze_site(site_code)
            
            if result and 'error' not in result:
                all_results.append(result)
                
                # Identify problematic sites for ecological modeling
                if result['eco_score'] < 70 or len(result['issues']) > 0:
                    problematic_sites.append(result)
        
        # Generate report
        self.generate_report(all_results, problematic_sites)
        
        # Summary
        if all_results:
            scores = [r['eco_score'] for r in all_results]
            print(f"\n✅ Analysis complete!")
            print(f"📊 {len(all_results)} sites analyzed")
            print(f"⚠️  {len(problematic_sites)} problematic for ecological modeling")
            print(f"📈 Average ecological score: {np.mean(scores):.1f}/100")
            print(f"📉 Range: {np.min(scores):.1f} to {np.max(scores):.1f}")
    
    def generate_report(self, all_results, problematic_sites):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = self.output_dir / f"ecological_quality_{timestamp}.txt"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ECOLOGICAL DATA QUALITY ANALYSIS\n")
            f.write("Focus: Variables critical for ecological modeling\n")
            f.write("Ignores: Timestamps and technical variables\n")
            f.write("=" * 50 + "\n\n")
            
            # Summary stats
            scores = [r['eco_score'] for r in all_results]
            f.write(f"Sites analyzed: {len(all_results)}\n")
            f.write(f"Average ecological score: {np.mean(scores):.1f}/100\n")
            f.write(f"Range: {np.min(scores):.1f} - {np.max(scores):.1f}\n\n")
            
            # Categorize sites
            excellent = [r for r in all_results if r['eco_score'] >= 85]
            good = [r for r in all_results if 70 <= r['eco_score'] < 85]
            problematic = [r for r in all_results if r['eco_score'] < 70]
            
            f.write("SITE CATEGORIES:\n")
            f.write(f"🟢 Excellent (≥85): {len(excellent)} sites\n")
            f.write(f"🟡 Good (70-84): {len(good)} sites\n") 
            f.write(f"🔴 Problematic (<70): {len(problematic)} sites\n\n")
            
            # Most problematic sites
            if problematic_sites:
                f.write("MOST PROBLEMATIC SITES FOR ECOLOGICAL MODELING:\n")
                f.write("-" * 45 + "\n")
                
                prob_sorted = sorted(problematic_sites, key=lambda x: x['eco_score'])
                for i, site in enumerate(prob_sorted[:20], 1):
                    f.write(f"\n{i:2d}. {site['site_code']} (Score: {site['eco_score']}/100)\n")
                    for issue in site['issues']:
                        f.write(f"    - {issue}\n")
            
            # Variable-specific analysis
            f.write(f"\n\nVARIABLE-SPECIFIC ZERO VALUE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            
            for var_code, var_info in self.critical_eco_vars.items():
                sites_with_issues = []
                for result in all_results:
                    if var_code in result['zero_stats'] and result['zero_stats'][var_code]['problematic']:
                        sites_with_issues.append((
                            result['site_code'], 
                            result['zero_stats'][var_code]['zero_pct']
                        ))
                
                sites_with_issues.sort(key=lambda x: x[1], reverse=True)
                
                f.write(f"\n{var_info['name']} - Sites with excessive zeros:\n")
                if sites_with_issues:
                    for i, (site, zero_pct) in enumerate(sites_with_issues[:10], 1):
                        f.write(f"  {i:2d}. {site}: {zero_pct:.1f}%\n")
                else:
                    f.write("  ✓ No problematic sites\n")
            
            # Best sites for modeling
            if excellent:
                f.write(f"\n\nBEST SITES FOR ECOLOGICAL MODELING:\n")
                f.write("-" * 35 + "\n")
                excellent_sorted = sorted(excellent, key=lambda x: x['eco_score'], reverse=True)
                for i, site in enumerate(excellent_sorted, 1):
                    f.write(f"{i:2d}. {site['site_code']}: {site['eco_score']}/100\n")
        
        print(f"📝 Report saved: {report_file}")

if __name__ == "__main__":
    analyzer = EcoQualityAnalyzer()
    analyzer.run_analysis() 