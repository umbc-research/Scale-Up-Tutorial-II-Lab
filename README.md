# K-Means Clustering Example

A simple, reproducible example of K-Means clustering using the Iris dataset. Designed as a teaching example for students learning about clustering algorithms and reproducible research practices.

## Overview

This project demonstrates:
- Loading and exploring the Iris dataset
- Exploratory Data Analysis (EDA) with visualizations
- Finding the optimal number of clusters using the **Elbow Method**
- Applying K-Means clustering with K=3
- Visual comparison of clustering results

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

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/K-Means-Clustering.git
cd K-Means-Clustering
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the script

```bash
python K_Means_Clustering.py
```

All output figures will be saved to the `output/` directory.

## Requirements

- Python 3.8+
- numpy
- matplotlib
- pandas
- seaborn
- scikit-learn

See `requirements.txt` for version details.

## HPC Cluster Usage

If running on an HPC cluster with SLURM (e.g., UMBC's HPCF):

```bash
sbatch run_kmeans.slurm
```

Edit `run_kmeans.slurm` to set your username and activate your Python virtual environment.

## What the Script Does

1. **Load Data**: Imports the Iris dataset using scikit-learn
2. **EDA**: Generates a pairplot showing feature relationships colored by species
3. **Elbow Method**: Computes Within-Cluster Sum of Squares (WCSS) for K=1 to 10
4. **Comparison**: Visualizes clustering results for K=1,2,3,4 alongside original labels
5. **Final Model**: Applies K-Means with K=3 (optimal based on elbow method)

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

## License

This project is open source and available under the MIT License.
