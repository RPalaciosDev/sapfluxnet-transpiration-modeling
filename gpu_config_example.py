# GPU Configuration for spatial_parquet.py
# Add these modifications to enable GPU acceleration

# 1. In the _validate_cluster_streaming_parquet method, modify xgb_params:

def _validate_cluster_streaming_parquet(self, cluster_id, train_file, test_file):
    """Modified for GPU acceleration"""
    
    # Original CPU parameters
    xgb_params_cpu = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',        # CPU histogram method
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 1
    }
    
    # GPU-accelerated parameters
    xgb_params_gpu = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'gpu_hist',    # GPU histogram method
        'gpu_id': 0,                  # Use first GPU
        'max_depth': 8,               # Can handle deeper trees on GPU
        'learning_rate': 0.15,        # Slightly higher learning rate
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'reg_alpha': 0.0,
        'reg_lambda': 1.0,
        'random_state': 42,
        'verbosity': 1
    }
    
    # Auto-detect GPU availability
    try:
        import xgboost as xgb
        if xgb.gpu.is_supported():
            print(f"🚀 Using GPU acceleration for cluster {cluster_id}")
            xgb_params = xgb_params_gpu
        else:
            print(f"💻 Using CPU for cluster {cluster_id}")
            xgb_params = xgb_params_cpu
    except:
        print(f"💻 GPU not available, using CPU for cluster {cluster_id}")
        xgb_params = xgb_params_cpu
    
    # Continue with existing training logic...
    dtrain = xgb.DMatrix(train_file)
    dtest = xgb.DMatrix(test_file)
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=100,  # Can increase with GPU
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=False
    )

# 2. For hyperparameter optimization in optimize_cluster_hyperparameters:

def optimize_cluster_hyperparameters(self, cluster_id, sample_train_file, sample_test_file):
    """Modified for GPU optimization"""
    
    def objective(trial):
        # GPU-optimized parameter ranges
        if self.use_gpu:
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'max_depth': trial.suggest_int('max_depth', 6, 12),      # Deeper trees
                'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 2.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 5.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 200),  # More trees
                'random_state': 42,
                'verbosity': 0
            }
        else:
            # Original CPU parameters (existing code)
            params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'tree_method': 'hist',
                'max_depth': trial.suggest_int('max_depth', 3, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
                'subsample': trial.suggest_float('subsample', 0.6, 0.9),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.9),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
                'n_estimators': trial.suggest_int('n_estimators', 50, 100),
                'random_state': 42,
                'n_jobs': -1,
                'verbosity': 0
            }
        
        # Rest of optimization logic...

# 3. Add GPU detection to __init__ method:

def __init__(self, ...):
    # Existing initialization...
    
    # Detect GPU availability
    try:
        import xgboost as xgb
        self.use_gpu = xgb.gpu.is_supported()
        if self.use_gpu:
            print("🚀 GPU acceleration available and enabled")
            print(f"💾 GPU memory optimization: tree_method='gpu_hist'")
        else:
            print("💻 Using CPU-only acceleration")
    except:
        self.use_gpu = False
        print("💻 GPU support not available, using CPU")