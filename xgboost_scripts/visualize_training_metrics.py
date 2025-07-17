#!/usr/bin/env python3
"""
Visualize detailed training metrics from external memory XGBoost training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import glob
from datetime import datetime
import argparse

def load_metrics_history(file_path):
    """Load metrics history from JSON or CSV file"""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
        return pd.DataFrame(data)
    elif file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    else:
        raise ValueError("File must be .json or .csv format")

def find_latest_metrics_file(directory='external_memory_models/random_split'):
    """Find the most recent metrics history file"""
    # Use the exact file path you have
    file_path = 'external_memory_models/random_split/sapfluxnet_external_memory_history.csv'
    
    if os.path.exists(file_path):
        print(f"Found metrics file: {file_path}")
        return file_path
    else:
        print(f"File not found: {file_path}")
        return None

def plot_training_curves(metrics_df, save_dir='feature_importance_plots'):
    """Plot comprehensive training curves"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('XGBoost Training Metrics Over Time', fontsize=16, fontweight='bold')
    
    # 1. R² Score progression
    axes[0, 0].plot(metrics_df['iteration'], metrics_df['train_r2'], label='Train R²', linewidth=2)
    axes[0, 0].plot(metrics_df['iteration'], metrics_df['test_r2'], label='Test R²', linewidth=2)
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Score Progression')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. RMSE progression
    axes[0, 1].plot(metrics_df['iteration'], metrics_df['train_rmse'], label='Train RMSE', linewidth=2)
    axes[0, 1].plot(metrics_df['iteration'], metrics_df['test_rmse'], label='Test RMSE', linewidth=2)
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Progression')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. MAE progression
    axes[0, 2].plot(metrics_df['iteration'], metrics_df['train_mae'], label='Train MAE', linewidth=2)
    axes[0, 2].plot(metrics_df['iteration'], metrics_df['test_mae'], label='Test MAE', linewidth=2)
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('MAE')
    axes[0, 2].set_title('MAE Progression')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Memory usage over time
    # Handle both column names for memory usage
    memory_col = 'memory_usage_gb' if 'memory_usage_gb' in metrics_df.columns else 'memory_usage'
    axes[1, 0].plot(metrics_df['iteration'], metrics_df[memory_col], 
                    color='purple', linewidth=2)
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Memory Usage (GB)')
    axes[1, 0].set_title('Memory Usage During Training')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Training vs Test gap (overfitting analysis)
    r2_gap = metrics_df['train_r2'] - metrics_df['test_r2']
    rmse_gap = metrics_df['test_rmse'] - metrics_df['train_rmse']
    
    axes[1, 1].plot(metrics_df['iteration'], r2_gap, label='R² Gap (Train-Test)', 
                    color='red', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].set_ylabel('R² Gap')
    axes[1, 1].set_title('Overfitting Analysis (R² Gap)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Convergence analysis
    # Calculate moving averages for smoother visualization
    window = min(10, len(metrics_df) // 4)
    if window > 1:
        train_r2_ma = metrics_df['train_r2'].rolling(window=window).mean()
        test_r2_ma = metrics_df['test_r2'].rolling(window=window).mean()
        
        axes[1, 2].plot(metrics_df['iteration'], train_r2_ma, 
                        label=f'Train R² (MA-{window})', linewidth=2)
        axes[1, 2].plot(metrics_df['iteration'], test_r2_ma, 
                        label=f'Test R² (MA-{window})', linewidth=2)
    else:
        axes[1, 2].plot(metrics_df['iteration'], metrics_df['train_r2'], 
                        label='Train R²', linewidth=2)
        axes[1, 2].plot(metrics_df['iteration'], metrics_df['test_r2'], 
                        label='Test R²', linewidth=2)
    
    axes[1, 2].set_xlabel('Iteration')
    axes[1, 2].set_ylabel('R² Score')
    axes[1, 2].set_title('Convergence Analysis (Smoothed)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved: {save_path}")
    plt.close()

def plot_performance_summary(metrics_df, save_dir='feature_importance_plots'):
    """Plot performance summary statistics"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Training Performance Summary', fontsize=16, fontweight='bold')
    
    # 1. Final performance comparison
    final_metrics = ['train_r2', 'test_r2', 'train_rmse', 'test_rmse', 'train_mae', 'test_mae']
    final_values = [metrics_df[col].iloc[-1] for col in final_metrics]
    
    x_pos = np.arange(len(final_metrics))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']
    
    bars = axes[0, 0].bar(x_pos, final_values, color=colors, alpha=0.7)
    axes[0, 0].set_xlabel('Metrics')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Final Model Performance')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(final_metrics, rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, final_values):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Training progression heatmap
    # Select key metrics for heatmap
    heatmap_data = metrics_df[['train_r2', 'test_r2', 'train_rmse', 'test_rmse']].T
    
    im = axes[0, 1].imshow(heatmap_data, cmap='RdYlBu_r', aspect='auto')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Metrics')
    axes[0, 1].set_title('Training Progression Heatmap')
    axes[0, 1].set_yticks(range(len(heatmap_data.index)))
    axes[0, 1].set_yticklabels(heatmap_data.index)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes[0, 1])
    cbar.set_label('Metric Value')
    
    # 3. Memory usage distribution
    # Handle both column names for memory usage
    memory_col = 'memory_usage_gb' if 'memory_usage_gb' in metrics_df.columns else 'memory_usage'
    axes[1, 0].hist(metrics_df[memory_col], bins=20, alpha=0.7, color='purple')
    axes[1, 0].axvline(metrics_df[memory_col].mean(), color='red', 
                       linestyle='--', label=f'Mean: {metrics_df[memory_col].mean():.2f}GB')
    axes[1, 0].set_xlabel('Memory Usage (GB)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Memory Usage Distribution')
    axes[1, 0].legend()
    
    # 4. Improvement over time
    # Calculate improvement from start to end
    r2_improvement = metrics_df['test_r2'].iloc[-1] - metrics_df['test_r2'].iloc[0]
    rmse_improvement = metrics_df['test_rmse'].iloc[0] - metrics_df['test_rmse'].iloc[-1]
    
    improvement_data = {
        'R² Improvement': r2_improvement,
        'RMSE Improvement': rmse_improvement,
        'Final R²': metrics_df['test_r2'].iloc[-1],
        'Final RMSE': metrics_df['test_rmse'].iloc[-1]
    }
    
    x_pos = np.arange(len(improvement_data))
    bars = axes[1, 1].bar(x_pos, list(improvement_data.values()), 
                          color=['green', 'blue', 'orange', 'red'], alpha=0.7)
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Value')
    axes[1, 1].set_title('Training Improvement Summary')
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(improvement_data.keys(), rotation=45)
    
    # Add value labels
    for bar, value in zip(bars, improvement_data.values()):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, 'performance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Performance summary saved: {save_path}")
    plt.close()

def plot_learning_curves(metrics_df, save_dir='feature_importance_plots'):
    """Plot detailed learning curves with multiple views"""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Learning Curves Analysis', fontsize=16, fontweight='bold')
    
    # 1. R² Learning Curve with best iteration marker
    axes[0, 0].plot(metrics_df['iteration'], metrics_df['train_r2'], 
                    label='Train R²', linewidth=2, color='blue', alpha=0.8)
    axes[0, 0].plot(metrics_df['iteration'], metrics_df['test_r2'], 
                    label='Test R²', linewidth=2, color='red', alpha=0.8)
    
    # Mark best iteration
    best_iter = metrics_df['test_r2'].idxmax()
    best_r2 = metrics_df.loc[best_iter, 'test_r2']
    axes[0, 0].axvline(x=best_iter, color='green', linestyle='--', alpha=0.7, 
                       label=f'Best Iteration ({best_iter})')
    axes[0, 0].scatter(best_iter, best_r2, color='green', s=100, zorder=5)
    
    axes[0, 0].set_xlabel('Training Iteration')
    axes[0, 0].set_ylabel('R² Score')
    axes[0, 0].set_title('R² Learning Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add final performance annotation
    final_train_r2 = metrics_df['train_r2'].iloc[-1]
    final_test_r2 = metrics_df['test_r2'].iloc[-1]
    axes[0, 0].text(0.02, 0.98, f'Final Train R²: {final_train_r2:.4f}\nFinal Test R²: {final_test_r2:.4f}', 
                    transform=axes[0, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 2. RMSE Learning Curve
    axes[0, 1].plot(metrics_df['iteration'], metrics_df['train_rmse'], 
                    label='Train RMSE', linewidth=2, color='blue', alpha=0.8)
    axes[0, 1].plot(metrics_df['iteration'], metrics_df['test_rmse'], 
                    label='Test RMSE', linewidth=2, color='red', alpha=0.8)
    
    # Mark best iteration for RMSE
    best_rmse_iter = metrics_df['test_rmse'].idxmin()
    best_rmse = metrics_df.loc[best_rmse_iter, 'test_rmse']
    axes[0, 1].axvline(x=best_rmse_iter, color='green', linestyle='--', alpha=0.7, 
                       label=f'Best Iteration ({best_rmse_iter})')
    axes[0, 1].scatter(best_rmse_iter, best_rmse, color='green', s=100, zorder=5)
    
    axes[0, 1].set_xlabel('Training Iteration')
    axes[0, 1].set_ylabel('RMSE')
    axes[0, 1].set_title('RMSE Learning Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add final performance annotation
    final_train_rmse = metrics_df['train_rmse'].iloc[-1]
    final_test_rmse = metrics_df['test_rmse'].iloc[-1]
    axes[0, 1].text(0.02, 0.98, f'Final Train RMSE: {final_train_rmse:.4f}\nFinal Test RMSE: {final_test_rmse:.4f}', 
                    transform=axes[0, 1].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. Validation Gap Analysis (Train vs Test gap)
    r2_gap = metrics_df['train_r2'] - metrics_df['test_r2']
    
    axes[1, 0].plot(metrics_df['iteration'], r2_gap, 
                    label='R² Gap (Train - Test)', linewidth=2, color='orange')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_xlabel('Training Iteration')
    axes[1, 0].set_ylabel('Performance Gap')
    axes[1, 0].set_title('Validation Gap Analysis')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add overfitting warning if gap is large
    final_gap = r2_gap.iloc[-1]
    if final_gap > 0.05:
        axes[1, 0].text(0.02, 0.02, f'⚠️ Potential overfitting\nFinal gap: {final_gap:.4f}', 
                        transform=axes[1, 0].transAxes, verticalalignment='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    else:
        axes[1, 0].text(0.02, 0.02, f'✅ Good generalization\nFinal gap: {final_gap:.4f}', 
                        transform=axes[1, 0].transAxes, verticalalignment='bottom',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
    
    # 4. Learning Rate Analysis (improvement per iteration)
    # Calculate rolling improvement
    window = min(20, len(metrics_df) // 10)
    if window > 1:
        test_r2_smooth = metrics_df['test_r2'].rolling(window=window, center=True).mean()
        improvement_rate = test_r2_smooth.diff().fillna(0)
        
        axes[1, 1].plot(metrics_df['iteration'], improvement_rate, 
                        linewidth=2, color='purple', alpha=0.8)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_xlabel('Training Iteration')
        axes[1, 1].set_ylabel('R² Improvement Rate')
        axes[1, 1].set_title(f'Learning Rate (R² change, {window}-iter window)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Find when learning plateaus
        plateau_threshold = 0.001
        recent_improvements = improvement_rate.tail(20)
        if recent_improvements.abs().mean() < plateau_threshold:
            axes[1, 1].text(0.02, 0.98, f'📊 Learning plateaued\nRecent avg improvement: {recent_improvements.mean():.6f}', 
                            transform=axes[1, 1].transAxes, verticalalignment='top',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
    else:
        axes[1, 1].text(0.5, 0.5, 'Not enough data\nfor learning rate analysis', 
                        transform=axes[1, 1].transAxes, ha='center', va='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    # Save the plot
    save_path = os.path.join(save_dir, 'learning_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Learning curves saved: {save_path}")
    plt.close()

def print_training_summary(metrics_df):
    """Print a comprehensive training summary"""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    print(f"Training Duration: {len(metrics_df)} iterations")
    
    # Handle different time column names
    if 'timestamp' in metrics_df.columns:
        print(f"Training Time: {metrics_df['timestamp'].iloc[-1] - metrics_df['timestamp'].iloc[0]:.1f} seconds")
    elif 'time_per_iteration' in metrics_df.columns:
        print(f"Total Training Time: {metrics_df['time_per_iteration'].sum():.1f} seconds")
    else:
        print("Training Time: Not available")
    
    print(f"\nFinal Performance:")
    print(f"  Train R²: {metrics_df['train_r2'].iloc[-1]:.4f}")
    print(f"  Test R²:  {metrics_df['test_r2'].iloc[-1]:.4f}")
    print(f"  Train RMSE: {metrics_df['train_rmse'].iloc[-1]:.4f}")
    print(f"  Test RMSE:  {metrics_df['test_rmse'].iloc[-1]:.4f}")
    print(f"  Train MAE: {metrics_df['train_mae'].iloc[-1]:.4f}")
    print(f"  Test MAE:  {metrics_df['test_mae'].iloc[-1]:.4f}")
    
    print(f"\nImprovement:")
    print(f"  R² Improvement: {metrics_df['test_r2'].iloc[-1] - metrics_df['test_r2'].iloc[0]:.4f}")
    print(f"  RMSE Improvement: {metrics_df['test_rmse'].iloc[0] - metrics_df['test_rmse'].iloc[-1]:.4f}")
    
    print(f"\nMemory Usage:")
    # Handle both column names for memory usage
    memory_col = 'memory_usage_gb' if 'memory_usage_gb' in metrics_df.columns else 'memory_usage'
    print(f"  Average: {metrics_df[memory_col].mean():.2f} GB")
    print(f"  Peak: {metrics_df[memory_col].max():.2f} GB")
    print(f"  Min: {metrics_df[memory_col].min():.2f} GB")
    
    # Find best iteration
    best_iter = metrics_df['test_r2'].idxmax()
    print(f"\nBest Performance (Iteration {best_iter}):")
    print(f"  Test R²: {metrics_df.loc[best_iter, 'test_r2']:.4f}")
    print(f"  Test RMSE: {metrics_df.loc[best_iter, 'test_rmse']:.4f}")

def main():
    """Main function to visualize training metrics"""
    parser = argparse.ArgumentParser(description='Visualize XGBoost training metrics')
    parser.add_argument('--file', type=str, help='Path to metrics history file (.json or .csv)')
    parser.add_argument('--directory', type=str, default='external_memory_models/random_split',
                        help='Directory to search for metrics files')
    parser.add_argument('--output-dir', type=str, default='feature_importance_plots',
                        help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Find metrics file
    if args.file:
        metrics_file = args.file
        if not os.path.exists(metrics_file):
            print(f"Error: File not found: {metrics_file}")
            return
    else:
        metrics_file = find_latest_metrics_file(args.directory)
        if not metrics_file:
            print("No metrics file found. Run training with --track-metrics first.")
            return
    
    print(f"Loading metrics from: {metrics_file}")
    
    try:
        # Load metrics data
        metrics_df = load_metrics_history(metrics_file)
        print(f"Loaded {len(metrics_df)} training iterations")
        
        # Create visualizations
        print("\nCreating training curves...")
        plot_training_curves(metrics_df, args.output_dir)
        
        print("Creating performance summary...")
        plot_performance_summary(metrics_df, args.output_dir)
        
        print("Creating learning curves...")
        plot_learning_curves(metrics_df, args.output_dir)
        
        # Print summary
        print_training_summary(metrics_df)
        
        print(f"\n✅ Visualizations completed!")
        print(f"📊 Plots saved in: {args.output_dir}")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 