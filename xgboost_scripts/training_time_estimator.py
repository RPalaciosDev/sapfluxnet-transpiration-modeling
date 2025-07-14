"""
SAPFLUXNET XGBoost Training Time Estimator
Pre-training assessment tool for estimating training time across all validation methods
"""

import dask.dataframe as dd
import pandas as pd
import numpy as np
import psutil
import os
import time
from datetime import datetime, timedelta
import platform
import subprocess
import warnings
warnings.filterwarnings('ignore')

class TrainingTimeEstimator:
    """Estimates training time for SAPFLUXNET XGBoost models based on data size and system specs"""
    
    def __init__(self, data_dir='comprehensive_processed'):
        self.data_dir = data_dir
        self.system_info = {}
        self.data_info = {}
        self.estimates = {}
        
    def get_system_info(self):
        """Collect comprehensive system information"""
        print("🔍 Analyzing system specifications...")
        
        # Basic system info
        self.system_info = {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        self.system_info.update({
            'total_memory_gb': memory.total / (1024**3),
            'available_memory_gb': memory.available / (1024**3),
            'memory_percent_used': memory.percent,
        })
        
        # CPU information
        self.system_info.update({
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_freq_max': psutil.cpu_freq().max if psutil.cpu_freq() else 'Unknown',
            'cpu_freq_current': psutil.cpu_freq().current if psutil.cpu_freq() else 'Unknown',
        })
        
        # Disk information
        disk_usage = psutil.disk_usage('.')
        self.system_info.update({
            'disk_total_gb': disk_usage.total / (1024**3),
            'disk_free_gb': disk_usage.free / (1024**3),
            'disk_used_percent': (disk_usage.used / disk_usage.total) * 100,
        })
        
        # Check if we're in Google Colab
        try:
            import google.colab
            self.system_info['environment'] = 'Google Colab'
            self.system_info['colab_type'] = 'Pro' if memory.total > 14 * (1024**3) else 'Free'
        except ImportError:
            self.system_info['environment'] = 'Local Machine'
        
        # GPU information (if available)
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                self.system_info['gpu_available'] = True
                self.system_info['gpu_count'] = len(gpus)
                self.system_info['gpu_memory_gb'] = gpus[0].memoryTotal / 1024
            else:
                self.system_info['gpu_available'] = False
        except ImportError:
            self.system_info['gpu_available'] = False
        
        print("✅ System analysis complete")
        return self.system_info
    
    def analyze_data_characteristics(self):
        """Analyze the dataset to understand training requirements"""
        print("📊 Analyzing dataset characteristics...")
        
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        # Find all processed files
        files = [f for f in os.listdir(self.data_dir) if f.endswith(('.parquet', '.csv'))]
        
        if not files:
            raise FileNotFoundError(f"No data files found in {self.data_dir}")
        
        print(f"  Found {len(files)} data files")
        
        # Sample a few files to understand structure
        sample_files = files[:min(5, len(files))]
        total_rows = 0
        total_size_mb = 0
        column_counts = []
        
        # Load sample to understand structure
        print("  Loading sample files to analyze structure...")
        
        start_time = time.time()
        for i, file in enumerate(sample_files):
            file_path = os.path.join(self.data_dir, file)
            file_size = os.path.getsize(file_path) / (1024**2)  # MB
            total_size_mb += file_size
            
            try:
                if file.endswith('.parquet'):
                    df_sample = pd.read_parquet(file_path, nrows=1000)
                else:
                    df_sample = pd.read_csv(file_path, nrows=1000)
                
                # Estimate total rows in this file
                if len(df_sample) == 1000:
                    # Estimate based on file size
                    estimated_rows = int((file_size * 1000) / (len(df_sample.memory_usage(deep=True).sum()) / (1024**2)))
                else:
                    estimated_rows = len(df_sample)
                
                total_rows += estimated_rows
                column_counts.append(len(df_sample.columns))
                
                if i == 0:
                    # Store structure info from first file
                    self.data_info['sample_columns'] = list(df_sample.columns)
                    self.data_info['has_target'] = 'sap_flow' in df_sample.columns
                    self.data_info['has_timestamp'] = any('timestamp' in col.lower() for col in df_sample.columns)
                    self.data_info['has_site'] = 'site' in df_sample.columns
                
            except Exception as e:
                print(f"  Warning: Could not analyze {file}: {e}")
                continue
        
        load_time = time.time() - start_time
        
        # Estimate total dataset size
        avg_file_size = total_size_mb / len(sample_files)
        estimated_total_size_mb = avg_file_size * len(files)
        estimated_total_rows = int((total_rows / len(sample_files)) * len(files))
        
        # Calculate loading performance
        rows_per_second = total_rows / load_time if load_time > 0 else 0
        mb_per_second = total_size_mb / load_time if load_time > 0 else 0
        
        self.data_info.update({
            'total_files': len(files),
            'sampled_files': len(sample_files),
            'estimated_total_rows': estimated_total_rows,
            'estimated_total_size_mb': estimated_total_size_mb,
            'estimated_total_size_gb': estimated_total_size_mb / 1024,
            'avg_columns_per_file': np.mean(column_counts) if column_counts else 0,
            'loading_performance': {
                'rows_per_second': rows_per_second,
                'mb_per_second': mb_per_second,
                'sample_load_time': load_time,
            }
        })
        
        # Estimate memory requirements
        if column_counts:
            avg_cols = np.mean(column_counts)
            # Rough estimate: 8 bytes per float64 value
            estimated_memory_per_row = avg_cols * 8 / (1024**2)  # MB per row
            estimated_peak_memory_gb = (estimated_total_rows * estimated_memory_per_row) / 1024
            
            self.data_info['memory_estimates'] = {
                'estimated_memory_per_row_mb': estimated_memory_per_row,
                'estimated_peak_memory_gb': estimated_peak_memory_gb,
                'memory_efficiency_needed': estimated_peak_memory_gb > self.system_info['available_memory_gb'] * 0.8
            }
        
        print(f"✅ Dataset analysis complete")
        print(f"  Estimated rows: {estimated_total_rows:,}")
        print(f"  Estimated size: {estimated_total_size_mb:.1f} MB ({estimated_total_size_mb/1024:.1f} GB)")
        print(f"  Average columns: {np.mean(column_counts):.0f}" if column_counts else "  Columns: Unknown")
        
        return self.data_info
    
    def estimate_training_times(self):
        """Estimate training time for each validation method"""
        print("⏱️  Calculating training time estimates...")
        
        if not self.data_info or not self.system_info:
            raise ValueError("Must run system and data analysis first")
        
        # Base factors for estimation
        rows = self.data_info['estimated_total_rows']
        size_gb = self.data_info['estimated_total_size_gb']
        cols = self.data_info['avg_columns_per_file']
        
        # System performance factors
        memory_gb = self.system_info['available_memory_gb']
        cpu_count = self.system_info['cpu_count_logical']
        
        # Environment-specific multipliers
        env_multiplier = 1.0
        if self.system_info['environment'] == 'Google Colab':
            env_multiplier = 1.2 if self.system_info.get('colab_type') == 'Free' else 1.0
        
        # Memory constraint multiplier
        memory_pressure = self.data_info['memory_estimates']['estimated_peak_memory_gb'] / memory_gb
        memory_multiplier = 1.0 + max(0, (memory_pressure - 0.5) * 2)  # Penalty if using >50% memory
        
        # Base training time estimation (empirical formula)
        # Based on: T = (rows * cols * boosting_rounds) / (cpu_performance * memory_bandwidth)
        base_seconds_per_million_features = 1.2  # Empirical constant
        
        def estimate_model_time(method_name, boosting_rounds, splits_multiplier, complexity_multiplier=1.0):
            """Estimate time for a specific validation method"""
            
            # Core training time
            million_features = (rows * cols) / 1_000_000
            base_training_time = million_features * base_seconds_per_million_features * boosting_rounds / 150
            
            # Apply multipliers
            total_time = (
                base_training_time * 
                splits_multiplier * 
                complexity_multiplier * 
                env_multiplier * 
                memory_multiplier
            )
            
            # Add overhead times
            data_loading_time = size_gb / max(1, self.data_info['loading_performance']['mb_per_second'] / 1024)
            preprocessing_time = rows / 50000  # ~50k rows per second preprocessing
            evaluation_time = rows / 100000   # ~100k rows per second evaluation
            
            total_time += data_loading_time + preprocessing_time + evaluation_time
            
            return {
                'total_minutes': total_time / 60,
                'breakdown': {
                    'data_loading_minutes': data_loading_time / 60,
                    'preprocessing_minutes': preprocessing_time / 60,
                    'training_minutes': (base_training_time * splits_multiplier * complexity_multiplier) / 60,
                    'evaluation_minutes': evaluation_time / 60,
                },
                'confidence': 'Medium' if memory_pressure < 0.8 else 'Low'
            }
        
        # Estimate each validation method
        self.estimates = {
            'random_split': estimate_model_time(
                'Random Split', 
                boosting_rounds=150, 
                splits_multiplier=1.0,  # Single split
                complexity_multiplier=1.0
            ),
            'k_fold_temporal': estimate_model_time(
                'K-Fold Temporal',
                boosting_rounds=150,
                splits_multiplier=5.0,  # 5 folds
                complexity_multiplier=1.2  # Temporal complexity
            ),
            'spatial_validation': estimate_model_time(
                'Spatial Validation',
                boosting_rounds=150,
                splits_multiplier=min(20, self.data_info['total_files']),  # Up to 20 sites
                complexity_multiplier=1.1  # Site exclusion complexity
            ),
            'rolling_window': estimate_model_time(
                'Rolling Window',
                boosting_rounds=100,  # Fewer rounds
                splits_multiplier=12,  # 12 windows
                complexity_multiplier=1.3  # Temporal windowing complexity
            )
        }
        
        print("✅ Training time estimates calculated")
        return self.estimates
    
    def check_memory_requirements(self):
        """Check if system has sufficient memory for training"""
        print("🧠 Checking memory requirements...")
        
        peak_memory_gb = self.data_info['memory_estimates']['estimated_peak_memory_gb']
        available_gb = self.system_info['available_memory_gb']
        
        recommendations = []
        
        if peak_memory_gb > available_gb * 0.9:
            recommendations.append("❌ Insufficient memory - training may fail")
            recommendations.append(f"   Need: {peak_memory_gb:.1f}GB, Available: {available_gb:.1f}GB")
            recommendations.append("   Recommend: Reduce dataset size or use more powerful machine")
        elif peak_memory_gb > available_gb * 0.7:
            recommendations.append("⚠️  High memory usage - training may be slow")
            recommendations.append(f"   Need: {peak_memory_gb:.1f}GB, Available: {available_gb:.1f}GB")
            recommendations.append("   Recommend: Close other applications")
        else:
            recommendations.append("✅ Sufficient memory available")
            recommendations.append(f"   Need: {peak_memory_gb:.1f}GB, Available: {available_gb:.1f}GB")
        
        return recommendations
    
    def print_system_summary(self):
        """Print comprehensive system summary"""
        print("\n" + "="*60)
        print("🖥️  SYSTEM SPECIFICATIONS")
        print("="*60)
        
        info = self.system_info
        print(f"Environment: {info['environment']}")
        if 'colab_type' in info:
            print(f"Colab Type: {info['colab_type']}")
        print(f"Platform: {info['platform']} {info['architecture']}")
        print(f"CPU: {info['cpu_count_logical']} logical cores ({info['cpu_count_physical']} physical)")
        print(f"Memory: {info['total_memory_gb']:.1f}GB total, {info['available_memory_gb']:.1f}GB available ({info['memory_percent_used']:.1f}% used)")
        print(f"Disk: {info['disk_free_gb']:.1f}GB free / {info['disk_total_gb']:.1f}GB total")
        if info['gpu_available']:
            print(f"GPU: Available ({info.get('gpu_count', 1)} GPUs, {info.get('gpu_memory_gb', 'Unknown')}GB)")
        else:
            print("GPU: Not available")
    
    def print_data_summary(self):
        """Print comprehensive data summary"""
        print("\n" + "="*60)
        print("📊 DATASET CHARACTERISTICS")
        print("="*60)
        
        info = self.data_info
        print(f"Files: {info['total_files']} files")
        print(f"Estimated Rows: {info['estimated_total_rows']:,}")
        print(f"Estimated Size: {info['estimated_total_size_gb']:.1f}GB")
        print(f"Average Columns: {info['avg_columns_per_file']:.0f}")
        print(f"Has Target Column: {'Yes' if info['has_target'] else 'No'}")
        print(f"Has Timestamp: {'Yes' if info['has_timestamp'] else 'No'}")
        print(f"Has Site Column: {'Yes' if info['has_site'] else 'No'}")
        
        # Loading performance
        perf = info['loading_performance']
        print(f"\nLoading Performance:")
        print(f"  {perf['rows_per_second']:,.0f} rows/second")
        print(f"  {perf['mb_per_second']:.1f} MB/second")
        
        # Memory estimates
        mem = info['memory_estimates']
        print(f"\nMemory Estimates:")
        print(f"  Peak Usage: {mem['estimated_peak_memory_gb']:.1f}GB")
        print(f"  Memory Efficient Mode: {'Required' if mem['memory_efficiency_needed'] else 'Optional'}")
    
    def print_time_estimates(self):
        """Print detailed training time estimates"""
        print("\n" + "="*60)
        print("⏱️  TRAINING TIME ESTIMATES")
        print("="*60)
        
        model_names = {
            'random_split': 'Random Split Baseline',
            'k_fold_temporal': 'K-Fold Temporal Validation',
            'spatial_validation': 'Spatial Validation (LOSO)',
            'rolling_window': 'Rolling Window Forecasting'
        }
        
        total_time = 0
        
        for method, name in model_names.items():
            est = self.estimates[method]
            minutes = est['total_minutes']
            total_time += minutes
            
            print(f"\n{name}:")
            print(f"  Total Time: {minutes:.1f} minutes ({minutes/60:.1f} hours)")
            print(f"  Confidence: {est['confidence']}")
            
            breakdown = est['breakdown']
            print(f"  Breakdown:")
            print(f"    Data Loading: {breakdown['data_loading_minutes']:.1f} min")
            print(f"    Preprocessing: {breakdown['preprocessing_minutes']:.1f} min")
            print(f"    Training: {breakdown['training_minutes']:.1f} min")
            print(f"    Evaluation: {breakdown['evaluation_minutes']:.1f} min")
        
        print(f"\n📊 SUMMARY:")
        print(f"Total Time (All Models): {total_time:.1f} minutes ({total_time/60:.1f} hours)")
        print(f"Recommended Order: Random → Temporal → Spatial → Rolling Window")
    
    def print_recommendations(self):
        """Print actionable recommendations"""
        print("\n" + "="*60)
        print("💡 RECOMMENDATIONS")
        print("="*60)
        
        memory_recs = self.check_memory_requirements()
        for rec in memory_recs:
            print(rec)
        
        print(f"\n🚀 Training Strategy:")
        
        # Memory-based recommendations
        if self.data_info['memory_estimates']['memory_efficiency_needed']:
            print("- Use memory-efficient mode (enabled by default)")
            print("- Consider reducing number of sites for spatial validation")
            print("- Close unnecessary applications before training")
        
        # Time-based recommendations
        fastest_time = min(est['total_minutes'] for est in self.estimates.values())
        if fastest_time > 60:
            print("- Consider running during off-peak hours")
            print("- Use Google Colab Pro for faster training")
        
        # Validation strategy
        print(f"\n📋 Suggested Validation Sequence:")
        print("1. Random Split (fastest) - Establish baseline")
        print("2. K-Fold Temporal - Test future prediction")
        print("3. Spatial Validation - Test site transferability")
        print("4. Rolling Window - Test operational forecasting")
        
        # Environment-specific tips
        if self.system_info['environment'] == 'Google Colab':
            print(f"\n☁️  Google Colab Tips:")
            print("- Keep browser tab active to prevent disconnection")
            print("- Save intermediate results frequently")
            print("- Consider upgrading to Colab Pro for longer sessions")
    
    def save_assessment_report(self, output_file='training_assessment_report.txt'):
        """Save complete assessment to file"""
        with open(output_file, 'w') as f:
            f.write("SAPFLUXNET XGBoost Training Assessment Report\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            
            # System info
            f.write("SYSTEM SPECIFICATIONS\n")
            f.write("-" * 25 + "\n")
            for key, value in self.system_info.items():
                f.write(f"{key}: {value}\n")
            
            # Data info
            f.write("\nDATASET CHARACTERISTICS\n")
            f.write("-" * 25 + "\n")
            for key, value in self.data_info.items():
                if isinstance(value, dict):
                    f.write(f"{key}:\n")
                    for subkey, subvalue in value.items():
                        f.write(f"  {subkey}: {subvalue}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            # Time estimates
            f.write("\nTRAINING TIME ESTIMATES\n")
            f.write("-" * 25 + "\n")
            for method, estimate in self.estimates.items():
                f.write(f"{method}: {estimate['total_minutes']:.1f} minutes\n")
                for key, value in estimate['breakdown'].items():
                    f.write(f"  {key}: {value:.1f} minutes\n")
        
        print(f"📝 Assessment report saved: {output_file}")
    
    def run_complete_assessment(self, save_report=True):
        """Run complete pre-training assessment"""
        print("🔍 SAPFLUXNET XGBoost Training Assessment")
        print("=" * 50)
        print(f"Started: {datetime.now()}")
        
        try:
            # Run all analyses
            self.get_system_info()
            self.analyze_data_characteristics()
            self.estimate_training_times()
            
            # Print comprehensive report
            self.print_system_summary()
            self.print_data_summary()
            self.print_time_estimates()
            self.print_recommendations()
            
            if save_report:
                self.save_assessment_report()
            
            print(f"\n✅ Assessment complete!")
            print(f"🎯 Ready to proceed with training based on estimates above")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Assessment failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main assessment function"""
    print("🔍 Pre-Training Assessment for SAPFLUXNET XGBoost Models")
    
    # Check if we're in Google Colab
    try:
        from google.colab import drive
        print("📱 Google Colab detected - mounting drive...")
        drive.mount('/content/drive')
        data_dir = '/content/drive/MyDrive/comprehensive_processed'
    except ImportError:
        print("💻 Local environment detected")
        data_dir = 'comprehensive_processed'
    
    # Run assessment
    estimator = TrainingTimeEstimator(data_dir)
    success = estimator.run_complete_assessment()
    
    if success:
        print(f"\n🚀 You can now proceed with training using the XGBoost scripts:")
        print(f"   python xgboost_scripts/random_xgboost.py")
        print(f"   python xgboost_scripts/temporal_validation_xgboost.py")
        print(f"   python xgboost_scripts/spatial_validation_XGBoost.py")
        print(f"   python xgboost_scripts/rolling_window_xgboost.py")
    else:
        print(f"\n⚠️  Please resolve issues before proceeding with training")

if __name__ == "__main__":
    main() 