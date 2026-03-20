# K-Means Clustering Example

A simple, reproducible example of K-Means clustering using the Iris dataset. Designed as a teaching example for students learning about clustering algorithms and reproducible research practices.

## Overview

This project demonstrates:
- Loading and exploring the Iris dataset
- Exploratory Data Analysis (EDA) with visualizations
- Finding the optimal number of clusters using the **Elbow Method**
- Verifying Elbow Method by K-Means clustering with K=[1, 2, 3, 4]
- Visual comparison of clustering results using different K values

## Project Structure

```
.
├── K_Means_Clustering.py   # Main Python script
├── requirements.txt        # Python dependencies
├── output/                 # Generated figures (created at runtime)
│   ├── pairplot.png
│   ├── elbow_method.png
│   └── kmeans_comparison.png
├── run_kmeans.slurm        # SLURM job script for HPC clusters
└── LICENSE
```

## Environment Setup

### 1. Clone the repository

```bash
git clone https://github.com/lixy4567/Scale-Up-Tutorial-II-Lab-Session.git
cd Scale-Up-Tutorial-II-Lab-Session
```

### 2. Install dependencies

For HPC use on chip cluster, start an interactive session by 'srun' and load the Python module

```bash
srun --cluster=chip-cpu --mem=5000 --time=1:30:00 --qos=normal --account=hpcf-scales --partition=general --pty $SHELL
module load Python/3.12.3-GCCcore-13.3.0
```

Create an PyVenv environment and instal dependencies

```bash
python -m venv /umbc/rs/hpcf-scales/users/${username}/KMeans
source /umbc/rs/hpcf-scales/users/${username}/KMeans/bin/activate
pip install -r requirements.txt
```


All output figures will be saved to the `output/` directory.

## Requirements

- Python 3.8+
- numpy
- matplotlib
- pandas
- seaborn
- scikit-learn

See `requirements.txt` for version details. See Install dependencies" for installation instructions.

## HPC Cluster Usage

If running on an HPC cluster with SLURM (e.g., UMBC's HPCF):

**Method 1:** Run it in an interactive session

```bash
srun --cluster=chip-cpu --mem=5000 --time=1:30:00 --qos=normal --account=hpcf-scales --partition=general --pty $SHELL
python K_Means_Clustering.py
```

**Method 2:** Use a Slurm script

Create a SLURM script to submit your job. Please refer to UMBC chip wiki tutorial on [Batch run using sbatch with a Slurm script](https://umbc.atlassian.net/wiki/spaces/faq/pages/1325957222/How+to+Run+on+chip#Batch-run-using-sbatch-with-a-Slurm-script).
 
Edit `run_kmeans.slurm` to load a Python module and activate your Python virtual environment. Please refer to [Tutorial II materials](https://docs.google.com/presentation/d/1ttjGDKAK_kmbxQ1E-RXfCSomNF48Coe9-N2AlGL4zpg/edit?usp=sharing), or [UMBC chip wiki tutorial on PyVenv](https://umbc.atlassian.net/wiki/spaces/faq/pages/1033863206/pyVenv+virtual+environments).

```bash
sbatch run_kmeans.slurm
```


## What the Script Does

1. **Load Data**: Imports the Iris dataset using scikit-learn
2. **EDA**: Generates a pairplot showing feature relationships colored by species
3. **Elbow Method**: Computes Within-Cluster Sum of Squares (WCSS) for K=1 to 10
4. **Comparison**: Visualizes clustering results for K=1,2,3,4 alongside original labels
5. **Final Model**: Confirms K-Means with K=3 is the optimal based on elbow method and comparison results

## Expected Output

After running, the `output/` directory will contain:
- `pairplot.png` - Feature pair relationships
- `elbow_method.png` - WCSS plot for finding optimal K
- `kmeans_comparison.png` - Side-by-side comparison of clustering results

## Reproducibility

This project is designed with reproducibility in mind:
- Fixed random state (`random_state=0`) in K-Means for deterministic results
- All dependencies are version-pinned
- Clear, documented code structure

## Attribution

The K-Means clustering implementation in this project is based on the
[Clustering Algorithms from Scratch](https://github.com/milaan9/Clustering_Algorithms_from_Scratch)
repository by [milaan9](https://github.com/milaan9), modified to run on HPC clusters
via SLURM job scheduling and extended with EDA visualizations.

## License

This project is open source and available under the MIT License.