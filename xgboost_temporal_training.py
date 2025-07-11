#!/usr/bin/env python3
"""
XGBoost Training with Multi-Scale Temporal Splitting for SAPFLUXNET Data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

class SAPFLUXNETTemporalTrainer:
    """Advanced XGBoost trainer with multi-scale temporal splitting strategies"""
    
    def __init__(self, data_dir='comprehensive_processed', output_dir='xgboost_models'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Sites with ≥3 years of data (high-value long-term datasets)
        self.long_term_sites = {
            'FRA_PUE', 'CAN_TUR_P39_PRE', 'USA_HIL_HF2', 'ESP_YUN_T3_THI',
            'ESP_YUN_C1', 'ESP_YUN_T1_THI', 'NZL_HUA_HUA', 'NLD_LOO',
            'ESP_TIL_MIX', 'ESP_YUN_C2', 'ESP_VAL_BAR', 'ESP_VAL_SOR', 'ESP_TIL_OAK'
        }
        
        # Sites with ≥2 years of data (moderate-term datasets)
        self.medium_term_sites = {
            'ARG_MAZ', 'ARG_TRE', 'AUS_BRI_BRI', 'AUS_CAN_ST1_EUC',
            'AUS_CAN_ST2_MIX', 'AUS_CAN_ST3_ACA', 'AUS_CAR_THI_00F',
            'AUS_CAR_THI_0PF', 'AUS_CAR_THI_CON', 'AUS_CAR_THI_T00',
            'AUS_CAR_THI_T0F', 'AUS_CAR_THI_TP0', 'AUS_CAR_THI_TPF',
            'AUS_ELL_MB_MOD', 'AUS_ELL_UNB', 'AUS_KAR', 'AUS_MAR_HSD_HIG',
            'AUS_MAR_HSW_HIG', 'AUS_MAR_MSD_MOD', 'AUS_MAR_MSW_MOD',
            'AUS_MAR_UBD', 'AUS_MAR_UBW', 'AUS_WOM', 'AUT_PAT_FOR',
            'AUT_PAT_KRU', 'AUT_PAT_TRE', 'AUT_TSC', 'BRA_CAM', 'BRA_SAN',
            'CHE_DAV_SEE', 'CHE_PFY_CON', 'CHE_PFY_IRR', 'CHN_ARG_GWD',
            'CHN_ARG_GWS', 'CHN_HOR_AFF', 'CHN_YIN_ST1', 'CHN_YIN_ST2_DRO',
            'CHN_YIN_ST3_DRO', 'CHN_YUN_YUN', 'COL_MAC_SAF_RAD', 'CRI_TAM_TOW',
            'CZE_LIZ_LES', 'ESP_MAJ_MAI', 'ESP_MAJ_NOR_LM1', 'ESP_RIN',
            'ESP_RON_PIL', 'ESP_SAN_A2_45I', 'ESP_SAN_A_45I', 'ESP_SAN_B2_100',
            'ESP_SAN_B_100', 'ESP_YUN_C1', 'ESP_YUN_T1_THI', 'FIN_PET',
            'FRA_HES_HE1_NON', 'FRA_PUE', 'GBR_ABE_PLO', 'GBR_DEV_CON',
            'GBR_DEV_DRO', 'GBR_GUI_ST1', 'GUF_GUY_GUY', 'HUN_SIK',
            'IDN_JAM_OIL', 'IDN_JAM_RUB', 'IDN_PON_STE', 'ITA_TOR',
            'JPN_EBE_HYB', 'MDG_SEM_TAL', 'MDG_YOU_SHO', 'MEX_COR_YP',
            'NLD_LOO', 'NLD_SPE_DOU', 'NZL_HUA_HUA', 'PRT_MIT', 'PRT_PIN',
            'RUS_CHE_LOW', 'RUS_CHE_Y4', 'SEN_SOU_IRR', 'SEN_SOU_POS',
            'SEN_SOU_PRE', 'THA_KHU', 'USA_CHE_ASP', 'USA_CHE_MAP',
            'USA_HIL_HF1_PRE', 'USA_HIL_HF2', 'USA_INM', 'USA_MOR_SF',
            'USA_NWH', 'USA_PER_PER', 'USA_SWH', 'USA_TNB', 'USA_TNO',
            'USA_TNP', 'USA_TNY', 'USA_WVF', 'ZAF_FRA_FRA', 'ZAF_NOO_E3_IRR',
            'ZAF_RAD', 'ZAF_SOU_SOU', 'ZAF_WEL_SOR'
        }
        
        self.models = {}
        self.results = {}
        
    def load_and_prepare_data(self, strategy='multi_scale'):
        """Load and prepare data based on temporal strategy"""
        
        print(f"🔍 Loading data with strategy: {strategy}")
        
        if strategy == 'long_term_only':
            # Use only sites with ≥3 years of data
            sites_to_load = self.long_term_sites
            print(f"📊 Loading {len(sites_to_load)} long-term sites (≥3 years)")
            
        elif strategy == 'medium_term_plus':
            # Use sites with ≥2 years of data
            sites_to_load = self.medium_term_sites
            print(f"📊 Loading {len(sites_to_load)} medium+ term sites (≥2 years)")
            
        elif strategy == 'multi_scale':
            # Use all available sites but weight by temporal coverage
            sites_to_load = self.medium_term_sites
            print(f"📊 Loading {len(sites_to_load)} sites with multi-scale weighting")
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Load data from processed files
        all_data = []
        
        for site in sites_to_load:
            file_path = f"{self.data_dir}/{site}_comprehensive.csv"
            if os.path.exists(file_path):
                try:
                    site_data = pd.read_csv(file_path)
                    if len(site_data) > 0:
                        all_data.append(site_data)
                        print(f"  ✅ {site}: {len(site_data)} rows")
                except Exception as e:
                    print(f"  ❌ {site}: Error loading - {str(e)}")
        
        if not all_data:
            raise ValueError("No data loaded!")
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        print(f"\n📊 Combined dataset: {len(combined_data):,} rows, {len(combined_data.columns)} columns")
        
        # Prepare features and target
        self.prepare_features_and_target(combined_data)
        
        return combined_data
    
    def prepare_features_and_target(self, data):
        """Prepare features and target variable"""
        
        # Remove non-feature columns
        exclude_cols = ['TIMESTAMP', 'solar_TIMESTAMP', 'site', 'plant_id']
        
        # Find target variable (sap_flow)
        if 'sap_flow' not in data.columns:
            raise ValueError("Target variable 'sap_flow' not found!")
        
        # Prepare features
        feature_cols = [col for col in data.columns if col not in exclude_cols + ['sap_flow']]
        
        # Remove any remaining non-numeric columns
        numeric_features = []
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                numeric_features.append(col)
            else:
                print(f"  ⚠️  Dropping non-numeric column: {col}")
        
        self.feature_cols = numeric_features
        self.target_col = 'sap_flow'
        
        print(f"🎯 Target variable: {self.target_col}")
        print(f"📈 Features: {len(self.feature_cols)} numeric features")
        
        # Check for missing values
        missing_target = data[self.target_col].isna().sum()
        missing_features = data[self.feature_cols].isna().sum().sum()
        
        print(f"📊 Missing values - Target: {missing_target}, Features: {missing_features}")
        
        # Remove rows with missing target
        data_clean = data.dropna(subset=[self.target_col])
        print(f"🧹 Cleaned dataset: {len(data_clean):,} rows")
        
        return data_clean
    
    def create_temporal_splits(self, data, strategy='progressive'):
        """Create temporal train/test splits"""
        
        print(f"\n⏰ Creating temporal splits with strategy: {strategy}")
        
        # Sort by timestamp
        if 'TIMESTAMP' in data.columns:
            data = data.sort_values('TIMESTAMP').reset_index(drop=True)
            print(f"📅 Data period: {data['TIMESTAMP'].min()} to {data['TIMESTAMP'].max()}")
        
        splits = {}
        
        if strategy == 'progressive':
            # Progressive temporal split (80% train, 20% test)
            split_idx = int(len(data) * 0.8)
            
            train_data = data.iloc[:split_idx]
            test_data = data.iloc[split_idx:]
            
            splits['train'] = train_data
            splits['test'] = test_data
            
            print(f"📊 Progressive split - Train: {len(train_data):,}, Test: {len(test_data):,}")
            
        elif strategy == 'time_series_cv':
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5, test_size=0.2)
            
            X = data[self.feature_cols]
            y = data[self.target_col]
            
            cv_splits = []
            for train_idx, test_idx in tscv.split(X):
                train_data = data.iloc[train_idx]
                test_data = data.iloc[test_idx]
                cv_splits.append((train_data, test_data))
            
            splits['cv_splits'] = cv_splits
            print(f"📊 Time series CV - {len(cv_splits)} splits created")
            
        elif strategy == 'site_based':
            # Split by sites (train on some sites, test on others)
            sites = data['site'].unique()
            train_sites = sites[:int(len(sites) * 0.8)]
            test_sites = sites[int(len(sites) * 0.8):]
            
            train_data = data[data['site'].isin(train_sites)]
            test_data = data[data['site'].isin(test_sites)]
            
            splits['train'] = train_data
            splits['test'] = test_data
            
            print(f"📊 Site-based split - Train sites: {len(train_sites)}, Test sites: {len(test_sites)}")
            print(f"📊 Train: {len(train_data):,}, Test: {len(test_data):,}")
            
        elif strategy == 'hybrid':
            # Hybrid approach: temporal split within each site
            train_data_list = []
            test_data_list = []
            
            for site in data['site'].unique():
                site_data = data[data['site'] == site].sort_values('TIMESTAMP')
                split_idx = int(len(site_data) * 0.8)
                
                train_data_list.append(site_data.iloc[:split_idx])
                test_data_list.append(site_data.iloc[split_idx:])
            
            train_data = pd.concat(train_data_list, ignore_index=True)
            test_data = pd.concat(test_data_list, ignore_index=True)
            
            splits['train'] = train_data
            splits['test'] = test_data
            
            print(f"📊 Hybrid split - Train: {len(train_data):,}, Test: {len(test_data):,}")
        
        return splits
    
    def train_xgboost_model(self, train_data, test_data, model_name='default'):
        """Train XGBoost model with hyperparameter tuning"""
        
        print(f"\n🚀 Training XGBoost model: {model_name}")
        
        # Prepare training data
        X_train = train_data[self.feature_cols]
        y_train = train_data[self.target_col]
        X_test = test_data[self.feature_cols]
        y_test = test_data[self.target_col]
        
        print(f"📊 Training set: {len(X_train):,} samples")
        print(f"📊 Test set: {len(X_test):,} samples")
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 1000,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        
        # Early stopping
        eval_set = [(X_train, y_train), (X_test, y_test)]
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=100
        )
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test)
        }
        
        print(f"\n📊 Model Performance:")
        print(f"  Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"  Test RMSE: {metrics['test_rmse']:.4f}")
        print(f"  Train MAE: {metrics['train_mae']:.4f}")
        print(f"  Test MAE: {metrics['test_mae']:.4f}")
        print(f"  Train R²: {metrics['train_r2']:.4f}")
        print(f"  Test R²: {metrics['test_r2']:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🏆 Top 10 Most Important Features:")
        for i, row in feature_importance.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Store results
        self.models[model_name] = model
        self.results[model_name] = {
            'metrics': metrics,
            'feature_importance': feature_importance,
            'predictions': {
                'train': y_pred_train,
                'test': y_pred_test
            },
            'actual': {
                'train': y_train,
                'test': y_test
            }
        }
        
        return model, metrics
    
    def cross_validate_model(self, data, cv_splits, model_name='cv_model'):
        """Perform cross-validation"""
        
        print(f"\n🔄 Performing cross-validation: {model_name}")
        
        cv_metrics = []
        
        for i, (train_data, test_data) in enumerate(cv_splits):
            print(f"\n📊 CV Fold {i+1}/{len(cv_splits)}")
            
            model, metrics = self.train_xgboost_model(
                train_data, test_data, 
                model_name=f"{model_name}_fold_{i+1}"
            )
            
            cv_metrics.append(metrics)
        
        # Aggregate CV results
        avg_metrics = {}
        for metric in ['train_rmse', 'test_rmse', 'train_mae', 'test_mae', 'train_r2', 'test_r2']:
            values = [m[metric] for m in cv_metrics]
            avg_metrics[f"cv_{metric}_mean"] = np.mean(values)
            avg_metrics[f"cv_{metric}_std"] = np.std(values)
        
        print(f"\n📊 Cross-Validation Results:")
        for metric in ['test_rmse', 'test_mae', 'test_r2']:
            mean_val = avg_metrics[f"cv_{metric}_mean"]
            std_val = avg_metrics[f"cv_{metric}_std"]
            print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        
        return avg_metrics
    
    def save_model_and_results(self, model_name='final_model'):
        """Save trained model and results"""
        
        if model_name not in self.models:
            print(f"❌ Model {model_name} not found!")
            return
        
        # Save model
        model_path = f"{self.output_dir}/{model_name}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[model_name], f)
        
        # Save results
        results_path = f"{self.output_dir}/{model_name}_results.pkl"
        with open(results_path, 'wb') as f:
            pickle.dump(self.results[model_name], f)
        
        # Save feature importance
        feature_importance = self.results[model_name]['feature_importance']
        feature_path = f"{self.output_dir}/{model_name}_feature_importance.csv"
        feature_importance.to_csv(feature_path, index=False)
        
        print(f"💾 Model and results saved:")
        print(f"  Model: {model_path}")
        print(f"  Results: {results_path}")
        print(f"  Features: {feature_path}")
    
    def plot_results(self, model_name='final_model'):
        """Plot model results"""
        
        if model_name not in self.results:
            print(f"❌ Results for {model_name} not found!")
            return
        
        results = self.results[model_name]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted (Test)
        axes[0, 0].scatter(results['actual']['test'], results['predictions']['test'], alpha=0.5)
        axes[0, 0].plot([results['actual']['test'].min(), results['actual']['test'].max()], 
                       [results['actual']['test'].min(), results['actual']['test'].max()], 'r--')
        axes[0, 0].set_xlabel('Actual Sap Flow')
        axes[0, 0].set_ylabel('Predicted Sap Flow')
        axes[0, 0].set_title('Test Set: Actual vs Predicted')
        axes[0, 0].grid(True)
        
        # 2. Residuals
        residuals = results['actual']['test'] - results['predictions']['test']
        axes[0, 1].scatter(results['predictions']['test'], residuals, alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Sap Flow')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True)
        
        # 3. Feature Importance
        top_features = results['feature_importance'].head(15)
        axes[1, 0].barh(range(len(top_features)), top_features['importance'])
        axes[1, 0].set_yticks(range(len(top_features)))
        axes[1, 0].set_yticklabels(top_features['feature'])
        axes[1, 0].set_xlabel('Feature Importance')
        axes[1, 0].set_title('Top 15 Feature Importance')
        
        # 4. Performance Metrics
        metrics = results['metrics']
        metric_names = ['Train RMSE', 'Test RMSE', 'Train MAE', 'Test MAE', 'Train R²', 'Test R²']
        metric_values = [metrics['train_rmse'], metrics['test_rmse'], 
                        metrics['train_mae'], metrics['test_mae'],
                        metrics['train_r2'], metrics['test_r2']]
        
        bars = axes[1, 1].bar(metric_names, metric_values)
        axes[1, 1].set_ylabel('Metric Value')
        axes[1, 1].set_title('Model Performance Metrics')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Color code bars
        for i, bar in enumerate(bars):
            if 'Test' in metric_names[i]:
                bar.set_color('orange')
            else:
                bar.set_color('blue')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = f"{self.output_dir}/{model_name}_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results plot saved: {plot_path}")
        
        plt.show()
    
    def run_comprehensive_training(self, strategy='multi_scale', split_strategy='progressive'):
        """Run comprehensive training with multiple approaches"""
        
        print("🚀 SAPFLUXNET XGBoost Training Pipeline")
        print("=" * 60)
        print(f"⏰ Started at: {datetime.now()}")
        
        # Load data
        data = self.load_and_prepare_data(strategy)
        
        # Create temporal splits
        splits = self.create_temporal_splits(data, split_strategy)
        
        # Train models based on split strategy
        if split_strategy == 'time_series_cv':
            # Cross-validation approach
            cv_metrics = self.cross_validate_model(data, splits['cv_splits'], 'cv_model')
            print(f"\n✅ Cross-validation completed")
            
        else:
            # Standard train/test approach
            train_data = splits['train']
            test_data = splits['test']
            
            model, metrics = self.train_xgboost_model(train_data, test_data, 'final_model')
            print(f"\n✅ Model training completed")
        
        # Save results
        self.save_model_and_results('final_model')
        
        # Plot results
        self.plot_results('final_model')
        
        print(f"\n⏰ Finished at: {datetime.now()}")
        print("🎉 Training pipeline completed successfully!")

def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train XGBoost models with temporal splitting')
    parser.add_argument('--strategy', default='multi_scale', 
                       choices=['long_term_only', 'medium_term_plus', 'multi_scale'],
                       help='Data loading strategy')
    parser.add_argument('--split-strategy', default='progressive',
                       choices=['progressive', 'time_series_cv', 'site_based', 'hybrid'],
                       help='Temporal splitting strategy')
    parser.add_argument('--data-dir', default='comprehensive_processed',
                       help='Directory with processed data')
    parser.add_argument('--output-dir', default='xgboost_models',
                       help='Output directory for models and results')
    
    args = parser.parse_args()
    
    # Create trainer and run
    trainer = SAPFLUXNETTemporalTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
    
    trainer.run_comprehensive_training(
        strategy=args.strategy,
        split_strategy=args.split_strategy
    )

if __name__ == "__main__":
    main() 