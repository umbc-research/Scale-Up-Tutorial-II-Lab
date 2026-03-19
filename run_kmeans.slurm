#!/bin/bash

#SBATCH --job-name=K-Means              # Job name
#SBATCH --output=slurm.out              # Output file name
#SBATCH --error=slurm.err               # Error file name
#SBATCH --cluster=chip-cpu              # HPC cluster name
#SBATCH --account=hpcf-scales           # HPC account
#SBATCH --partition=general             # Partition/queue
#SBATCH --qos=short                     # QoS
#SBATCH --time=00:05:00                 # Time limit
#SBATCH --mem=4G                        # Memory requested
#SBATCH --nodes=1                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Tasks per node

# Load Python module and activate your virtual environment
# Replace /path/to/your/venv with your actual virtual environment path
module load Python/3.12.3-GCCcore-13.3.0
source ~/PythonEnv/bin/activate

echo "Activating Python environment"
which python

# Run the K-Means clustering script
python K_Means_Clustering.py
