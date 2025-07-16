"""
Training Metrics Visualization for SAPFLUXNET XGBoost Models
Analyzes and visualizes training progression, learning curves, and performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
import re
from datetime import datetime

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def find_latest_history_file(results_dir='xgboost_scripts/external_memory_models/random_split'):
    """Find the most recent training history file"""
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all history files
    history_files = [f for f in os.listdir(results_dir) if f.endswith('_history.csv')]
    
    if not history_files:
        raise FileNotFoundError(f"No history files found in {results_dir}")
    
    # Get the most recent file
    latest_file = max(history_files, key=lambda f: os.path.getctime(os.path.join(results_dir, f)))
    return os.path.join(results_dir, latest_file)

def load_training_history(file_path):
    """Load training history data"""
    print(f"Loading training history from: {file_path}")
    
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} training iterations")
    
    # Show sample of data
    print("\nSample of training history:")
    print(df.head())
    
    return df

def create_learning_curves(df, output_dir='training_metrics_plots'):
    """Create comprehensive learning curve visualizations"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. R² Learning Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(df['iteration'], df['train_r2'], label='Train R²', linewidth=2, color='blue')
    ax.plot(df['iteration'], df['test_r2'], label='Test R²', linewidth=2, color='red')
    
    # Add best iteration marker
    best_iter = df.loc[df['test_r2'].idxmax(), 'iteration']
    best_r2 = df['test_r2'].max()
    ax.axvline(x=best_iter, color='green', linestyle='--', alpha=0.7, label=f'Best Iteration ({best_iter})')
    ax.scatter(best_iter, best_r2, color='green', s=100, zorder=5)
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Learning Curves: R² Score Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add final values annotation
    final_train_r2 = df['train_r2'].iloc[-1]
    final_test_r2 = df['test_r2'].iloc[-1]
    ax.text(0.02, 0.98, f'Final Train R²: {final_train_r2:.4f}\nFinal Test R²: {final_test_r2:.4f}', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves_r2.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE Learning Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(df['iteration'], df['train_rmse'], label='Train RMSE', linewidth=2, color='blue')
    ax.plot(df['iteration'], df['test_rmse'], label='Test RMSE', linewidth=2, color='red')
    
    # Add best iteration marker
    best_iter_rmse = df.loc[df['test_rmse'].idxmin(), 'iteration']
    best_rmse = df['test_rmse'].min()
    ax.axvline(x=best_iter_rmse, color='green', linestyle='--', alpha=0.7, label=f'Best Iteration ({best_iter_rmse})')
    ax.scatter(best_iter_rmse, best_rmse, color='green', s=100, zorder=5)
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('RMSE', fontsize=12)
    ax.set_title('Learning Curves: RMSE Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add final values annotation
    final_train_rmse = df['train_rmse'].iloc[-1]
    final_test_rmse = df['test_rmse'].iloc[-1]
    ax.text(0.02, 0.98, f'Final Train RMSE: {final_train_rmse:.4f}\nFinal Test RMSE: {final_test_rmse:.4f}', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves_rmse.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. MAE Learning Curves
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(df['iteration'], df['train_mae'], label='Train MAE', linewidth=2, color='blue')
    ax.plot(df['iteration'], df['test_mae'], label='Test MAE', linewidth=2, color='red')
    
    # Add best iteration marker
    best_iter_mae = df.loc[df['test_mae'].idxmin(), 'iteration']
    best_mae = df['test_mae'].min()
    ax.axvline(x=best_iter_mae, color='green', linestyle='--', alpha=0.7, label=f'Best Iteration ({best_iter_mae})')
    ax.scatter(best_iter_mae, best_mae, color='green', s=100, zorder=5)
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Learning Curves: MAE Over Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add final values annotation
    final_train_mae = df['train_mae'].iloc[-1]
    final_test_mae = df['test_mae'].iloc[-1]
    ax.text(0.02, 0.98, f'Final Train MAE: {final_train_mae:.4f}\nFinal Test MAE: {final_test_mae:.4f}', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/learning_curves_mae.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Memory Usage Over Time
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(df['iteration'], df['memory_usage'], linewidth=2, color='purple')
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Memory Usage (GB)', fontsize=12)
    ax.set_title('Memory Usage During Training', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    max_memory = df['memory_usage'].max()
    avg_memory = df['memory_usage'].mean()
    ax.text(0.02, 0.98, f'Max Memory: {max_memory:.1f} GB\nAvg Memory: {avg_memory:.1f} GB', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/memory_usage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Training Time Analysis
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    ax.plot(df['iteration'], df['time_per_iteration'], linewidth=2, color='orange')
    
    ax.set_xlabel('Training Iteration', fontsize=12)
    ax.set_ylabel('Time per Iteration (seconds)', fontsize=12)
    ax.set_title('Training Time per Iteration', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    avg_time = df['time_per_iteration'].mean()
    total_time = df['time_per_iteration'].sum()
    ax.text(0.02, 0.98, f'Avg Time/Iter: {avg_time:.2f}s\nTotal Time: {total_time:.1f}s', 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Comprehensive Metrics Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R²
    axes[0, 0].plot(df['iteration'], df['train_r2'], label='Train', color='blue')
    axes[0, 0].plot(df['iteration'], df['test_r2'], label='Test', color='red')
    axes[0, 0].set_title('R² Score', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # RMSE
    axes[0, 1].plot(df['iteration'], df['train_rmse'], label='Train', color='blue')
    axes[0, 1].plot(df['iteration'], df['test_rmse'], label='Test', color='red')
    axes[0, 1].set_title('RMSE', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Memory Usage
    axes[1, 0].plot(df['iteration'], df['memory_usage'], color='purple')
    axes[1, 0].set_title('Memory Usage (GB)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Training Time
    axes[1, 1].plot(df['iteration'], df['time_per_iteration'], color='orange')
    axes[1, 1].set_title('Time per Iteration (s)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/training_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'best_iteration_r2': best_iter,
        'best_r2': best_r2,
        'final_train_r2': final_train_r2,
        'final_test_r2': final_test_r2,
        'total_training_time': total_time,
        'max_memory_usage': max_memory
    }

def create_summary_report(df, metrics_summary, output_dir='training_metrics_plots'):
    """Create a text summary report of training metrics"""
    
    report_path = f'{output_dir}/training_metrics_summary.txt'
    
    with open(report_path, 'w') as f:
        f.write("SAPFLUXNET Training Metrics Analysis Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Analysis completed: {datetime.now()}\n")
        f.write(f"Total training iterations: {len(df)}\n\n")
        
        f.write("KEY TRAINING METRICS:\n")
        f.write("-" * 20 + "\n")
        f.write(f"• Best iteration (R²): {metrics_summary['best_iteration_r2']}\n")
        f.write(f"• Best test R²: {metrics_summary['best_r2']:.4f}\n")
        f.write(f"• Final train R²: {metrics_summary['final_train_r2']:.4f}\n")
        f.write(f"• Final test R²: {metrics_summary['final_test_r2']:.4f}\n")
        f.write(f"• Total training time: {metrics_summary['total_training_time']:.1f} seconds\n")
        f.write(f"• Max memory usage: {metrics_summary['max_memory_usage']:.1f} GB\n\n")
        
        f.write("TRAINING CONVERGENCE ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        
        # Analyze convergence
        final_10_r2 = df['test_r2'].tail(10)
        r2_std = final_10_r2.std()
        r2_improvement = df['test_r2'].iloc[-1] - df['test_r2'].iloc[-10]
        
        f.write(f"• Final 10 iterations R² std: {r2_std:.6f}\n")
        f.write(f"• R² improvement in last 10 iterations: {r2_improvement:.6f}\n")
        
        if r2_std < 0.001:
            f.write("• Training appears to have converged (low variance in final iterations)\n")
        else:
            f.write("• Training may not have fully converged (high variance in final iterations)\n")
        
        if r2_improvement > 0.001:
            f.write("• Model was still improving in final iterations\n")
        else:
            f.write("• Model performance plateaued in final iterations\n")
        
        f.write("\nMEMORY AND PERFORMANCE INSIGHTS:\n")
        f.write("-" * 35 + "\n")
        f.write(f"• Average time per iteration: {df['time_per_iteration'].mean():.2f} seconds\n")
        f.write(f"• Average memory usage: {df['memory_usage'].mean():.1f} GB\n")
        f.write(f"• Memory efficiency: {metrics_summary['max_memory_usage']:.1f} GB peak usage\n")
        
        # Check for overfitting
        train_test_gap = metrics_summary['final_train_r2'] - metrics_summary['final_test_r2']
        f.write(f"• Train-test R² gap: {train_test_gap:.4f}\n")
        
        if train_test_gap > 0.05:
            f.write("• Potential overfitting detected (large train-test gap)\n")
        elif train_test_gap < 0.01:
            f.write("• Good generalization (small train-test gap)\n")
        else:
            f.write("• Moderate generalization (moderate train-test gap)\n")
        
        f.write(f"\nThis analysis helps understand the training dynamics and model convergence\n")
        f.write(f"for the SAPFLUXNET external memory XGBoost model.\n")
    
    print(f"Training metrics summary saved to: {report_path}")

def main():
    """Main function to run training metrics analysis"""
    
    print("SAPFLUXNET Training Metrics Visualization")
    print("=" * 50)
    
    try:
        # Find and load the latest history file
        history_file = find_latest_history_file()
        df = load_training_history(history_file)
        
        # Create visualizations
        print("\nCreating training metrics visualizations...")
        metrics_summary = create_learning_curves(df)
        
        # Create summary report
        print("Creating summary report...")
        create_summary_report(df, metrics_summary)
        
        print("\n✅ Training metrics analysis completed!")
        print("📊 Visualizations saved to: training_metrics_plots/")
        print("📝 Summary report: training_metrics_plots/training_metrics_summary.txt")
        
        print(f"\nQuick Summary:")
        print(f"  • Total iterations: {len(df)}")
        print(f"  • Best test R²: {metrics_summary['best_r2']:.4f}")
        print(f"  • Final test R²: {metrics_summary['final_test_r2']:.4f}")
        print(f"  • Training time: {metrics_summary['total_training_time']:.1f}s")
        print(f"  • Peak memory: {metrics_summary['max_memory_usage']:.1f}GB")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 