#!/usr/bin/env python3
"""
Site Splitting Script for Ensemble Validation
Creates reproducible 80/20 train/test split using same filtering as data pipeline
"""

import pandas as pd
import numpy as np
import glob
import os
import json
import argparse
from datetime import datetime

class SiteSplitter:
    """
    Creates reproducible train/test site splits for ensemble validation
    Uses same site discovery and filtering logic as data_pipeline_v3.py
    """
    
    def __init__(self, test_fraction=0.2, random_seed=42, output_file='site_split_assignment.json'):
        self.test_fraction = test_fraction
        self.random_seed = random_seed
        self.output_file = output_file
        
        # Same problematic site lists as data_pipeline_v3.py
        self.EXTREMELY_PROBLEMATIC_SITES = {
            'IDN_PON_STE',  # 63.1% flag rate - Extremely poor quality
            'USA_NWH',  # 53.4% flag rate - Very poor quality
        }
        
        self.HIGH_PROBLEMATIC_SITES = {
            'ZAF_NOO_E3_IRR',  # 25.9% flag rate - Very poor quality  
            'GUF_GUY_GUY',  # 35.5% flag rate - Very poor quality
            'USA_TNP',  # 31.6% flag rate - Very poor quality
            'USA_TNY',  # 28.9% flag rate - Very poor quality
            'USA_WVF',  # 16.6% flag rate - Very poor quality
        }
        
        self.MODERATE_PROBLEMATIC_SITES = {
            'USA_SYL_HL2',  # 16.0% flag rate - Poor quality
            'USA_WIL_WC2',  # 13.3% flag rate - Poor quality
            'CAN_TUR_P39_POS',  # 13.2% flag rate - Poor quality
            'CAN_TUR_P74',  # 15.8% flag rate - Poor quality
            'USA_PAR_FER',  # 16.7% flag rate - Poor quality
            'USA_TNB',  # 19.4% flag rate - Poor quality
            'USA_TNO',  # 19.3% flag rate - Poor quality
            'USA_UMB_GIR',  # 27.9% flag rate - Poor quality
            'FRA_PUE',  # 9.1% flag rate - Moderate issues
            'CAN_TUR_P39_PRE',  # 9.2% flag rate - Moderate issues
            'FRA_HES_HE2_NON',  # 9.0% flag rate - Moderate issues
            'USA_DUK_HAR',  # 6.0% flag rate - Moderate issues
            'USA_UMB_CON',  # 1.6% flag rate - Moderate issues
            'USA_PJS_P12_AMB',  # 3.0% flag rate - Moderate issues
            'USA_SIL_OAK_2PR',  # 3.0% flag rate - Moderate issues
            'USA_SIL_OAK_1PR',  # 3.0% flag rate - Moderate issues
            'USA_PJS_P04_AMB',  # 2.2% flag rate - Moderate issues
            'USA_PJS_P08_AMB',  # 1.8% flag rate - Moderate issues
            'USA_SIL_OAK_POS',  # 3.5% flag rate - Moderate issues
        }
        
        # Sites with no valid data or insufficient temporal coverage
        # These will be determined dynamically, but we include known ones
        self.SITES_WITH_NO_VALID_DATA = {
            'AUS_CAN_ST2_MIX', 'AUS_CAR_THI_0P0', 'AUS_CAR_THI_TP0', 'AUS_CAR_THI_TPF', 
            'AUS_ELL_HB_HIG', 'AUS_RIC_EUC_ELE', 'CAN_TUR_P39_POS', 'CAN_TUR_P74',
            'CHE_LOT_NOR', 'DEU_HIN_OAK', 'DEU_HIN_TER', 'DEU_STE_2P3', 'DEU_STE_4P5',
            'ESP_CAN', 'ESP_GUA_VAL', 'ESP_TIL_PIN', 'ESP_TIL_OAK', 'FIN_HYY_SME', 
            'FIN_PET', 'FRA_FON', 'FRA_HES_HE2_NON', 'GBR_GUI_ST2', 'GBR_GUI_ST3', 
            'GBR_DEV_CON', 'GBR_DEV_DRO', 'GUF_GUY_ST2', 'GUF_NOU_PET', 'JPN_EBE_SUG', 
            'JPN_EBE_HYB', 'KOR_TAE_TC1_LOW', 'KOR_TAE_TC2_MED', 'KOR_TAE_TC3_EXT',
            'MEX_VER_BSJ', 'MEX_VER_BSM', 'PRT_LEZ_ARN', 'RUS_FYO', 'RUS_CHE_Y4',
            'SWE_NOR_ST1_AF1', 'SWE_NOR_ST1_AF2', 'SWE_NOR_ST1_BEF', 'SWE_NOR_ST2',
            'SWE_NOR_ST3', 'SWE_NOR_ST4_AFT', 'SWE_NOR_ST4_BEF', 'SWE_NOR_ST5_REF',
            'SWE_SKO_MIN', 'SWE_SKY_38Y', 'SWE_SKY_68Y', 'USA_BNZ_BLA', 'USA_DUK_HAR', 
            'USA_HIL_HF1_POS', 'USA_HUY_LIN_NON', 'USA_PAR_FER', 'USA_PJS_P04_AMB', 
            'USA_PJS_P08_AMB', 'USA_PJS_P12_AMB', 'USA_SIL_OAK_1PR', 'USA_SIL_OAK_2PR', 
            'USA_SIL_OAK_POS', 'USA_SMI_SCB', 'USA_SMI_SER', 'USA_SYL_HL1', 'USA_SYL_HL2',
            'USA_UMB_CON', 'USA_UMB_GIR', 'USA_WIL_WC1', 'USA_WIL_WC2', 'USA_HIL_HF2', 'UZB_YAN_DIS'
        }
        
        self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE = {
            'ARG_MAZ',      # 12.0 days
            'ARG_TRE',      # 13.0 days  
            'COL_MAC_SAF_RAD'  # 13.2 days
        }
        
        print(f"🎯 Site Splitter initialized")
        print(f"📊 Test fraction: {test_fraction} ({test_fraction*100}% for testing)")
        print(f"🎲 Random seed: {random_seed}")
        print(f"📁 Output file: {output_file}")
    
    def get_all_sites(self):
        """Get all sites from sapwood directory (same logic as data_pipeline_v3.py)"""
        all_files = glob.glob('sapwood/*.csv')
        sites = set()
        
        for file in all_files:
            filename = os.path.basename(file)
            parts = filename.split('_')
            
            if len(parts) >= 2:
                if len(parts) == 4 and parts[2] in ['env', 'sapf', 'plant', 'site', 'species', 'stand']:
                    site = f"{parts[0]}_{parts[1]}"
                    sites.add(site)
                elif len(parts) >= 5 and parts[-2] in ['env', 'sapf', 'plant', 'site', 'species', 'stand']:
                    site = '_'.join(parts[:-2])
                    sites.add(site)
        
        return sorted(list(sites))
    
    def should_skip_site(self, site):
        """Determine if site should be skipped (same logic as data_pipeline_v3.py)"""
        # Sites with no valid data or insufficient temporal coverage
        if site in self.SITES_WITH_NO_VALID_DATA:
            return True, "No valid data"
        
        if site in self.SITES_WITH_INSUFFICIENT_TEMPORAL_COVERAGE:
            return True, "Insufficient temporal coverage (<30 days)"
        
        # Extremely problematic sites (always excluded)
        if site in self.EXTREMELY_PROBLEMATIC_SITES:
            return True, "Extremely problematic (>80% flag rate)"
        
        # High problematic sites (excluded in normal mode)
        if site in self.HIGH_PROBLEMATIC_SITES:
            return True, "High problematic (50-80% flag rate)"
        
        return False, None
    
    def validate_site_files(self, site):
        """Validate that site has required env and sapf files"""
        env_file = f'sapwood/{site}_env_data.csv'
        sapf_file = f'sapwood/{site}_sapf_data.csv'
        
        if not os.path.exists(env_file):
            return False, f"Missing environmental data file: {env_file}"
        
        if not os.path.exists(sapf_file):
            return False, f"Missing sap flow data file: {sapf_file}"
        
        return True, None
    
    def create_site_split(self):
        """Create the 80/20 train/test site split"""
        print(f"\n🔍 Discovering sites...")
        
        # Get all available sites
        all_sites = self.get_all_sites()
        print(f"  📁 Found {len(all_sites)} total sites in sapwood directory")
        
        # Filter sites using same logic as data pipeline
        valid_sites = []
        excluded_sites = {}
        
        for site in all_sites:
            # Check if site should be skipped based on quality
            should_skip, skip_reason = self.should_skip_site(site)
            if should_skip:
                excluded_sites[site] = skip_reason
                continue
            
            # Validate that required files exist
            files_exist, file_error = self.validate_site_files(site)
            if not files_exist:
                excluded_sites[site] = file_error
                continue
            
            valid_sites.append(site)
        
        print(f"  ✅ {len(valid_sites)} valid sites for splitting")
        print(f"  🚫 {len(excluded_sites)} sites excluded")
        
        if len(excluded_sites) > 0:
            print(f"    📋 Exclusion reasons:")
            exclusion_counts = {}
            for site, reason in excluded_sites.items():
                category = reason.split(':')[0] if ':' in reason else reason
                if category not in exclusion_counts:
                    exclusion_counts[category] = []
                exclusion_counts[category].append(site)
            
            for reason, sites in exclusion_counts.items():
                print(f"      - {reason}: {len(sites)} sites")
        
        if len(valid_sites) < 10:
            raise ValueError(f"Too few valid sites ({len(valid_sites)}) for meaningful train/test split")
        
        # Create reproducible random split
        print(f"\n🎲 Creating {(1-self.test_fraction)*100:.0f}/{self.test_fraction*100:.0f} train/test split...")
        
        np.random.seed(self.random_seed)
        shuffled_sites = valid_sites.copy()
        np.random.shuffle(shuffled_sites)
        
        split_idx = int(len(shuffled_sites) * (1 - self.test_fraction))
        train_sites = shuffled_sites[:split_idx]
        test_sites = shuffled_sites[split_idx:]
        
        print(f"  📊 Train sites: {len(train_sites)} ({len(train_sites)/len(valid_sites)*100:.1f}%)")
        print(f"  📊 Test sites: {len(test_sites)} ({len(test_sites)/len(valid_sites)*100:.1f}%)")
        
        # Show sample sites for verification
        train_sample = train_sites[:5] if len(train_sites) >= 5 else train_sites
        test_sample = test_sites[:5] if len(test_sites) >= 5 else test_sites
        print(f"  📋 Train sample: {train_sample}")
        print(f"  📋 Test sample: {test_sample}")
        
        return train_sites, test_sites, excluded_sites
    
    def save_site_split(self, train_sites, test_sites, excluded_sites):
        """Save the site split to JSON file"""
        print(f"\n💾 Saving site split...")
        
        split_data = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'test_fraction': self.test_fraction,
                'random_seed': self.random_seed,
                'total_valid_sites': len(train_sites) + len(test_sites),
                'train_count': len(train_sites),
                'test_count': len(test_sites),
                'excluded_count': len(excluded_sites)
            },
            'train_sites': sorted(train_sites),
            'test_sites': sorted(test_sites),
            'excluded_sites': dict(sorted(excluded_sites.items()))
        }
        
        with open(self.output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"  ✅ Site split saved to: {self.output_file}")
        print(f"  📊 {len(train_sites)} train sites, {len(test_sites)} test sites")
        print(f"  🚫 {len(excluded_sites)} excluded sites")
        
        return self.output_file
    
    def run(self):
        """Run the complete site splitting process"""
        print(f"🚀 SITE SPLITTING FOR ENSEMBLE VALIDATION")
        print(f"{'='*60}")
        print(f"Started at: {datetime.now()}")
        
        try:
            # Create the site split
            train_sites, test_sites, excluded_sites = self.create_site_split()
            
            # Save to file
            output_file = self.save_site_split(train_sites, test_sites, excluded_sites)
            
            print(f"\n🎉 Site splitting completed successfully!")
            print(f"📁 Split saved to: {output_file}")
            print(f"🎯 Ready for ensemble validation pipeline")
            
            return output_file
            
        except Exception as e:
            print(f"\n❌ Site splitting failed: {e}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Create reproducible train/test site split for ensemble validation")
    parser.add_argument('--test-fraction', type=float, default=0.2,
                        help="Fraction of sites to use for testing (default: 0.2)")
    parser.add_argument('--random-seed', type=int, default=42,
                        help="Random seed for reproducible splitting (default: 42)")
    parser.add_argument('--output-file', default='site_split_assignment.json',
                        help="Output file for site assignments (default: site_split_assignment.json)")
    
    args = parser.parse_args()
    
    print("🚀 SITE SPLITTING FOR ENSEMBLE VALIDATION")
    print("=" * 50)
    print(f"Test fraction: {args.test_fraction}")
    print(f"Random seed: {args.random_seed}")
    print(f"Output file: {args.output_file}")
    print(f"Started at: {datetime.now()}")
    
    try:
        splitter = SiteSplitter(
            test_fraction=args.test_fraction,
            random_seed=args.random_seed,
            output_file=args.output_file
        )
        
        output_file = splitter.run()
        
        print(f"\n✅ Site splitting completed successfully!")
        print(f"📁 Results: {output_file}")
        
    except Exception as e:
        print(f"\n❌ Site splitting failed: {e}")
        raise
    
    print(f"\nFinished at: {datetime.now()}")

if __name__ == "__main__":
    main() 