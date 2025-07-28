#!/bin/bash
#SBATCH --job-name=xgb_hyperopt_clusters
#SBATCH --account=your_account_name          # Replace with your SLURM account
#SBATCH --partition=cpu                      # Use CPU partition (adjust if needed)
#SBATCH --nodes=1                           # Single node job
#SBATCH --ntasks=1                          # Single task
#SBATCH --cpus-per-task=16                  # 16 CPU cores for parallel optimization
#SBATCH --mem=8G                            # 8GB RAM (conservative estimate)
#SBATCH --time=02:30:00                     # 2.5 hours (buffer included)
#SBATCH --output=hyperopt_%j.out            # Output file with job ID
#SBATCH --error=hyperopt_%j.err             # Error file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL          # Email notifications
#SBATCH --mail-user=your_email@domain.com   # Replace with your email

# Job information
echo "=========================================="
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Memory per node: $SLURM_MEM_PER_NODE"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Started at: $(date)"
echo "=========================================="

# Environment setup
module purge                                # Clear all modules
module load python/3.9                     # Load Python (adjust version as needed)
module load gcc/9.3.0                      # Load GCC compiler
# module load cuda/11.8                    # Uncomment if GPU available (not needed for this job)

# Activate your conda/virtual environment
# Option 1: Conda environment
# source activate your_env_name

# Option 2: Virtual environment
# source /path/to/your/venv/bin/activate

# Option 3: Module-based Python environment
# module load python-packages/your-package-set

echo "Python version: $(python --version)"
echo "Python path: $(which python)"

# Check available resources
echo "Available CPUs: $(nproc)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Current directory: $(pwd)"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Verify required Python packages
echo "Checking Python dependencies..."
python -c "
import sys
required_packages = ['pandas', 'numpy', 'xgboost', 'scikit-learn', 'optuna', 'psutil']
missing = []
for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        missing.append(pkg)
        print(f'✗ {pkg} - MISSING')

if missing:
    print(f'ERROR: Missing packages: {missing}')
    sys.exit(1)
else:
    print('All required packages found!')
"

# Check if the script exists
if [ ! -f "hyperparameter_optimization_hpc.py" ]; then
    echo "ERROR: hyperparameter_optimization_hpc.py not found in current directory"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

# Check if data directory exists
if [ ! -d "../../processed_parquet" ]; then
    echo "ERROR: processed_parquet directory not found"
    echo "Please ensure the parquet data files are accessible"
    exit 1
fi

echo "Starting hyperparameter optimization..."
echo "=========================================="

# Run the optimization with explicit parameters
python hyperparameter_optimization_hpc.py \
    --n-trials 100 \
    --n-jobs $SLURM_CPUS_PER_TASK \
    --parquet-dir ../../processed_parquet

# Capture exit code
exit_code=$?

echo "=========================================="
echo "Job completed at: $(date)"
echo "Exit code: $exit_code"

# Display resource usage
echo "Resource usage summary:"
if command -v sacct &> /dev/null; then
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,MaxRSS,Elapsed,State
else
    echo "sacct not available for resource summary"
fi

# Check if results were generated
if [ -d "results/hyperparameter_optimization" ]; then
    echo "Results directory contents:"
    ls -la results/hyperparameter_optimization/
    
    # Count generated files
    json_files=$(find results/hyperparameter_optimization/ -name "*.json" | wc -l)
    csv_files=$(find results/hyperparameter_optimization/ -name "*.csv" | wc -l)
    py_files=$(find results/hyperparameter_optimization/ -name "*.py" | wc -l)
    
    echo "Generated files: $json_files JSON, $csv_files CSV, $py_files Python"
else
    echo "WARNING: No results directory found"
fi

if [ $exit_code -eq 0 ]; then
    echo "✅ Hyperparameter optimization completed successfully!"
else
    echo "❌ Hyperparameter optimization failed with exit code: $exit_code"
fi

exit $exit_code 