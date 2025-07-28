# SLURM Hyperparameter Optimization Setup Guide

## 🎯 Quick Start

### 1. **Estimate Resources First**

```bash
cd ecosystem/models
python estimate_resources.py
```

This will analyze your data and provide specific SLURM resource recommendations.

### 2. **Customize SLURM Script**

Edit `submit_hyperopt.slurm` and update:

- `#SBATCH --account=your_account_name` → Your SLURM account
- `#SBATCH --partition=cpu` → Your cluster's CPU partition name
- `#SBATCH --mail-user=your_email@domain.com` → Your email
- Module names to match your cluster

### 3. **Submit Job**

```bash
sbatch submit_hyperopt.slurm
```

---

## 📋 Detailed Setup Instructions

### **Step 1: Check Your Cluster Setup**

First, find out your cluster's configuration:

```bash
# Check available partitions
sinfo

# Check your account
sacctmgr show user $USER

# Check available modules
module avail python
module avail gcc
```

### **Step 2: Customize Resource Requirements**

Based on your `estimate_resources.py` output, choose one:

#### **Conservative (Recommended for first run):**

```bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G  
#SBATCH --time=02:00:00
```

#### **Aggressive (If resources available):**

```bash
#SBATCH --cpus-per-task=24
#SBATCH --mem=12G
#SBATCH --time=01:30:00
```

#### **Minimal (Budget-friendly):**

```bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=4G
#SBATCH --time=04:00:00
```

### **Step 3: Environment Setup Options**

Choose the environment setup that matches your cluster:

#### **Option A: Conda Environment**

```bash
# In submit_hyperopt.slurm, uncomment:
source activate your_env_name
```

#### **Option B: Virtual Environment**

```bash
# In submit_hyperopt.slurm, uncomment:
source /path/to/your/venv/bin/activate
```

#### **Option C: Module-based**

```bash
# In submit_hyperopt.slurm, uncomment:
module load python-packages/your-package-set
```

### **Step 4: Required Python Packages**

Ensure these packages are installed in your environment:

- `pandas`
- `numpy`
- `xgboost`
- `scikit-learn`
- `optuna`
- `psutil`

---

## 🚀 Submission Commands

### **Standard Submission**

```bash
sbatch submit_hyperopt.slurm
```

### **Test Run (Fewer trials)**

```bash
# Edit the script to use --n-trials 20 for testing
sbatch submit_hyperopt.slurm
```

### **Monitor Job**

```bash
# Check job status
squeue -u $USER

# View output in real-time
tail -f hyperopt_JOBID.out

# Check resource usage
sstat -j JOBID --format=AveCPU,AveRSS,MaxRSS
```

---

## 📊 Expected Output Files

After successful completion, you'll find:

```
results/hyperparameter_optimization/
├── hpc_hyperparameter_optimization_TIMESTAMP.json    # Detailed results
├── hpc_hyperparameter_summary_TIMESTAMP.csv          # Summary table
├── hpc_optimized_spatial_params_TIMESTAMP.py         # Ready-to-use parameters
└── hpc_resource_summary_TIMESTAMP.txt                # Resource usage report
```

---

## 🔧 Troubleshooting

### **Common Issues:**

#### **1. Job Fails Immediately**

```bash
# Check error file
cat hyperopt_JOBID.err

# Common causes:
# - Wrong partition name
# - Invalid account
# - Missing modules
# - Environment not activated
```

#### **2. Out of Memory**

```bash
# Increase memory in SLURM script:
#SBATCH --mem=12G

# Or reduce data sampling in the Python script:
# max_sites=3, max_samples_per_site=5000
```

#### **3. Time Limit Exceeded**

```bash
# Increase time limit:
#SBATCH --time=04:00:00

# Or reduce trials:
# --n-trials 50
```

#### **4. Module Load Errors**

```bash
# Check available modules:
module avail python

# Update module names in submit_hyperopt.slurm
```

---

## ⚡ Performance Optimization Tips

### **1. Reduce Job Time**

- Start with `--n-trials 50` for testing
- Use more CPUs if available (`--cpus-per-task=24`)
- Consider reducing `max_sites=3` in the Python script

### **2. Save Resources**

- Use minimal memory settings for budget clusters
- Run during off-peak hours
- Consider splitting into separate jobs per cluster

### **3. Monitor Efficiency**

```bash
# After job completion, check efficiency:
sacct -j JOBID --format=JobID,CPUTime,TotalCPU,CPUTimeRAW,Elapsed,MaxRSS

# Good efficiency: TotalCPU ≈ CPUTime × 80%
```

---

## 📞 Getting Help

### **Cluster-Specific Help**

```bash
# Most clusters have documentation:
man sbatch
sinfo --help

# Contact your HPC support team for:
# - Account setup
# - Partition names  
# - Module availability
# - Resource limits
```

### **Job Debugging**

```bash
# Detailed job information:
scontrol show job JOBID

# Historical job data:
sacct -j JOBID --long

# Node information:
scontrol show node NODENAME
```

---

## 🎯 Success Criteria

Your job succeeded if:

- ✅ Exit code is 0
- ✅ All 3 clusters were optimized
- ✅ Results files were generated
- ✅ No "FAILED" or "TIMEOUT" in job status

Next step: Use the generated `hpc_optimized_spatial_params_TIMESTAMP.py` file to update your spatial validation!
