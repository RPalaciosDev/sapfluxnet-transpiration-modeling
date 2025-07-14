"""
Visualization utilities for SAPFLUXNET XGBoost models
Provides functions to create comprehensive visualizations and dashboards
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime
import pickle
import json

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def save_predictions_with_metadata(y_actual, y_pred, ddf_test, output_dir, model_type, timestamp):
    """Save predictions with site and temporal metadata for visualization"""
    
    # Extract metadata if available
    test_sample = ddf_test.compute() if hasattr(ddf_test, 'compute') else ddf_test
    
    predictions_df = pd.DataFrame({
        'actual': y_actual.flatten() if hasattr(y_actual, 'flatten') else y_actual,
        'predicted': y_pred.flatten() if hasattr(y_pred, 'flatten') else y_pred,
    })
    
    # Add metadata if available
    if 'site' in test_sample.columns:
        predictions_df['site'] = test_sample['site'].values[:len(predictions_df)]
    if 'TIMESTAMP' in test_sample.columns:
        predictions_df['timestamp'] = test_sample['TIMESTAMP'].values[:len(predictions_df)]
    
    # Save predictions
    pred_path = f"{output_dir}/predictions_{model_type}_{timestamp}.csv"
    predictions_df.to_csv(pred_path, index=False)
    
    return predictions_df, pred_path

def create_performance_plot(metrics, model_type, output_dir, timestamp):
    """Create basic performance visualization"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # R² comparison
    r2_data = [metrics['train_r2'], metrics['test_r2']]
    ax1.bar(['Train', 'Test'], r2_data, color=['blue', 'orange'], alpha=0.7)
    ax1.set_title(f'{model_type} - R² Score')
    ax1.set_ylabel('R² Score')
    ax1.set_ylim(0, 1)
    
    # RMSE comparison
    rmse_data = [metrics['train_rmse'], metrics['test_rmse']]
    ax2.bar(['Train', 'Test'], rmse_data, color=['blue', 'orange'], alpha=0.7)
    ax2.set_title(f'{model_type} - RMSE')
    ax2.set_ylabel('RMSE')
    
    # MAE comparison
    mae_data = [metrics['train_mae'], metrics['test_mae']]
    ax3.bar(['Train', 'Test'], mae_data, color=['blue', 'orange'], alpha=0.7)
    ax3.set_title(f'{model_type} - MAE')
    ax3.set_ylabel('MAE')
    
    # Performance summary text
    ax4.text(0.1, 0.8, f"Model: {model_type}", fontsize=14, weight='bold')
    ax4.text(0.1, 0.6, f"Test R²: {metrics['test_r2']:.4f}", fontsize=12)
    ax4.text(0.1, 0.4, f"Test RMSE: {metrics['test_rmse']:.4f}", fontsize=12)
    ax4.text(0.1, 0.2, f"Test MAE: {metrics['test_mae']:.4f}", fontsize=12)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = f"{output_dir}/performance_{model_type}_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_prediction_scatter(predictions_df, model_type, output_dir, timestamp):
    """Create actual vs predicted scatter plot"""
    
    plt.figure(figsize=(10, 8))
    
    # Scatter plot
    plt.scatter(predictions_df['actual'], predictions_df['predicted'], 
                alpha=0.5, s=20, color='blue')
    
    # Perfect prediction line
    min_val = min(predictions_df['actual'].min(), predictions_df['predicted'].min())
    max_val = max(predictions_df['actual'].max(), predictions_df['predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = np.corrcoef(predictions_df['actual'], predictions_df['predicted'])[0, 1]**2
    
    plt.xlabel('Actual Sap Flow')
    plt.ylabel('Predicted Sap Flow')
    plt.title(f'{model_type} - Actual vs Predicted')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add R² annotation
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save plot
    scatter_path = f"{output_dir}/scatter_{model_type}_{timestamp}.png"
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return scatter_path

def create_feature_importance_plot(feature_importance, model_type, output_dir, timestamp, top_n=15):
    """Create feature importance visualization"""
    
    # Get top N features
    top_features = feature_importance.head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    # Horizontal bar plot
    plt.barh(range(len(top_features)), top_features['importance'], 
             color='skyblue', alpha=0.8)
    
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score')
    plt.title(f'{model_type} - Top {top_n} Feature Importance')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save plot
    importance_path = f"{output_dir}/feature_importance_{model_type}_{timestamp}.png"
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_path

def create_residual_plots(predictions_df, model_type, output_dir, timestamp):
    """Create residual analysis plots"""
    
    residuals = predictions_df['predicted'] - predictions_df['actual']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Residuals vs Predicted
    ax1.scatter(predictions_df['predicted'], residuals, alpha=0.5, s=20)
    ax1.axhline(y=0, color='r', linestyle='--', lw=2)
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')
    ax1.set_title(f'{model_type} - Residuals vs Predicted')
    ax1.grid(True, alpha=0.3)
    
    # Residual distribution
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
    ax2.set_xlabel('Residuals')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{model_type} - Residual Distribution')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    residual_path = f"{output_dir}/residuals_{model_type}_{timestamp}.png"
    plt.savefig(residual_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return residual_path

def generate_model_visualizations(model, metrics, feature_importance, y_actual, y_pred, 
                                ddf_test, model_type, output_dir, timestamp):
    """Generate complete set of visualizations for a model"""
    
    print(f"Generating visualizations for {model_type}...")
    
    # Save predictions with metadata
    predictions_df, pred_path = save_predictions_with_metadata(
        y_actual, y_pred, ddf_test, output_dir, model_type, timestamp
    )
    
    # Create visualizations
    plots = {}
    
    # 1. Performance metrics
    plots['performance'] = create_performance_plot(metrics, model_type, output_dir, timestamp)
    
    # 2. Prediction scatter
    plots['scatter'] = create_prediction_scatter(predictions_df, model_type, output_dir, timestamp)
    
    # 3. Feature importance
    plots['importance'] = create_feature_importance_plot(feature_importance, model_type, output_dir, timestamp)
    
    # 4. Residual analysis
    plots['residuals'] = create_residual_plots(predictions_df, model_type, output_dir, timestamp)
    
    # Save visualization summary
    viz_summary = {
        'model_type': model_type,
        'timestamp': timestamp,
        'predictions_file': pred_path,
        'plots': plots,
        'metrics': metrics
    }
    
    summary_path = f"{output_dir}/visualization_summary_{model_type}_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(viz_summary, f, indent=2)
    
    print(f"✅ Visualizations saved:")
    for plot_type, path in plots.items():
        print(f"  {plot_type}: {path}")
    
    return viz_summary

def load_and_compare_models(results_dir='colab_models'):
    """Load results from multiple models and create comparison visualizations"""
    
    # Find all result files
    result_files = []
    for filename in os.listdir(results_dir):
        if filename.startswith('visualization_summary_') and filename.endswith('.json'):
            result_files.append(os.path.join(results_dir, filename))
    
    if not result_files:
        print("No model results found for comparison")
        return
    
    # Load all results
    all_results = []
    for file_path in result_files:
        with open(file_path, 'r') as f:
            results = json.load(f)
            all_results.append(results)
    
    # Create comparison plot
    model_types = [r['model_type'] for r in all_results]
    test_r2 = [r['metrics']['test_r2'] for r in all_results]
    test_rmse = [r['metrics']['test_rmse'] for r in all_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # R² comparison
    bars1 = ax1.bar(model_types, test_r2, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax1.set_title('Model Comparison - R² Score')
    ax1.set_ylabel('Test R² Score')
    ax1.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars1, test_r2):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # RMSE comparison
    bars2 = ax2.bar(model_types, test_rmse, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax2.set_title('Model Comparison - RMSE')
    ax2.set_ylabel('Test RMSE')
    
    # Add value labels on bars
    for bar, value in zip(bars2, test_rmse):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = f"{results_dir}/model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Model comparison saved: {comparison_path}")
    
    return comparison_path 