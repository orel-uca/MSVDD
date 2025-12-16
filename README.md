# MSVDD: Multi-Sphere Support Vector Data Description

An implementation of anomaly detection models based on Support Vector Data Description (SVDD) with support for multiple spheres and kernels.

## 📋 Description

This project implements advanced anomaly detection algorithms based on Support Vector Data Description (SVDD). It provides solutions for both single-sphere (SVDD) and multi-sphere (MSVDD) problems.

### Key Features

- **Classic SVDD**: Standard Support Vector Data Description implementation.
- **MSVDD**: Multi-sphere extension for anomaly detection.
- **Kernels**: Support for linear, RBF (Radial Basis Function), and polynomial kernels.
- **Synthetic and Real Data**: Support for synthetic and real datasets (ionosphere, iris, segment)

## 🚀 Installation

### Prerequisites

- Python 3.7+
- NumPy
- SciPy
- Gurobi 12.0 or later (for optimization)

### Installing Dependencies

```bash
# Create a conda environment
conda create -n gurobipy12 python=3.8

# Activate the environment
conda activate gurobipy12

# Install dependencies
pip install numpy scipy gurobi matplotlib scikit-learn
```## 📁 Project Structure

```

MSVDD/
├── README.md                  # This file
├── LICENSE                        # GNU GPL v3 License
├── msvdd_class.py            # Classes for instances and solutions
├── msvdd_models.py        # Optimization models (Primal and Dual)
├── utilities.py                      # Helper functions and visualization
├── test_MSVDD.py             # Test script for synthetic data
├── test_MSVDD_RDB.py    # Test script for real data
├── Test.ipynb                      # Notebook with usage examples
├── Data/                               # Synthetic datasets
└── RDB/                               # Real datasets
    ├── Data_ionosphere/
    ├── Data_iris/
    └── Data_segment/

## 🔧 Usage

### Example 1: Evaluation on synthetic data

```bash
python test_MSVDD.py \
  --dir_data "Data" \
  --dir_sufix "test" \
  --do_evaluation \
  --do_metrics \
  --reps 1 2 3 4 5 \
  --ks 1 2 3 r \
  --Cs 0.11 0.151 \
  --num_train 100 \
  --num_val 66 \
  --num_test 166 \
  --anom_frac 0.05 0.10
```

### Example 2: Evaluation on real data with RBF kernel

```bash
python test_MSVDD_RDB.py \
  --dir_data "RDB/Data_ionosphere" \
  --dir_sufix "ionosphere" \
  --do_evaluation \
  --do_metrics \
  --do_cross_val \
  --reps 1 2 3 4 5 \
  --ks 1 2 3\
  --Cs 0.070 0.075 \
  --sigmas 0.05 0.1 \
  --num_train 71 75 \
  --num_val 46 48 \
  --num_test 119 124 \
  --anom_frac 0.05 0.10
```

### 📊 Main Parameters

### Instance (Model Configuration)

- `data`: Data array (n_samples, n_features)
- `ntrain`: Number of training samples
- `nval`: Number of validation samples
- `ntest`: Number of test samples
- `y`: Labels (normal=0, anomaly=1)
- `p`: Number of clusters/spheres (default: 1)
- `C`: Regularization parameter (default: 0.1)
- `kernel`: Kernel type ('linear', 'rbf', 'polynomial')
- `sigma`: Parameter for RBF kernel
- `degree`: Degree for polynomial kernel

### Solution (Results)

- `c`: Cluster centers (p, d)
- `R`: Cluster radii (p,)
- `xi`: Slack variables (n,)
- `z`: Assignment variables (n, p)
- `alpha`: Dual coefficients (n, p)
- `obj`: Solution objective value
- `runtime`: Execution time
- `gap`: Optimality gap (MIP gap)

## 📈 Metrics

The project includes automatic calculation of:

- **AUC (Area Under the Curve)**: Overall classification performance
- **ROC curves**: Receiver Operating Characteristic curves
- **Cross-validation**: Robust model validation

## 📚 Main Functions

### utilities.py

- `Kernel()`: Kernel matrix computation
- `predict_scores()`: Score prediction for new points
- `DrawBalls()`: Model spheres visualization
- `DrawCurves()`: Decision boundary visualization
- `generate_data()`: Synthetic data generation
- `evaluate()`: MSVDD model evaluation
- `metrics()`: Metric calculation for MSVDD

### msvdd_models.py

- `Primal_MSVDD()`: Primal solution of MSVDD problem
- `Dualized_MSVDD()`: Kernelized solution of MSVDD problem

## 🔄 Typical Workflow

1. **Data Preparation**: Load or generate data
2. **Instance Creation**: Configure parameters
3. **Training**: Run optimizer (Primal or Dual)
4. **Evaluation**: Calculate metrics
5. **Cross-Validation**: Search for optimal hyperparameters
6. **Visualization**: Plot results

## 🧪 Included Datasets

### Synthetic

- Control variables: training, validation, and test set size, anomaly fraction
- Location: `Data/`

### Real

- **Ionosphere**: Ionospheric radar classification
- **Iris**: Iris flower classification
- **Segment**: Image segmentation
- Location: `RDB/`

## 📝 Command Line Arguments

### test_MSVDD.py and test_MSVDD_RDB.py

```
--dir_data              Data directory (default: "Data")
--dir_sufix_out         Suffix for output directories
--new_data              Generate new synthetic data (only available for test_MSVDD.py script)
--anom_frac             Anomaly fraction [0.05, 0.10, ...]
--num_train             Training set size
--num_val               Validation set size (auto-calculated if not specified)
--num_test              Test set size (auto-calculated if not specified)
--use_kernel            Use RBF kernel
--do_evaluation         Evaluate MSVDD models
--do_metrics            Calculate AUC-ROC for MSVDD
--do_cross_val          MSVDD cross-validation
--reps                  Number of repetitions
--ks                    K values
--Cs                    C parameter values
--sigmas                Sigma values for RBF kernel
--show_plots            Show plots (only available for test_MSVDD.py script)
```

## 🔗 References

- Based on the work of Nico Görnitz: [ClusterSvdd](https://github.com/nicococo/ClusterSvdd)
- Support Vector Data Description (SVDD): "*D. M. Tax, R. P. Duin, Support vector data description, Machine learning 54
  (2004) 45–66*".
- ClusterSVDD: "*N. Görnitz, L. A. Lima, K.-R. M¨uller, M. Kloft, S. Nakajima, Support vector data descriptions and k-means clustering: one class?, IEEE transactions on neural networks and learning systems 29 (9) (2017) 3994–4006*".
- Multisphere Support Vector Data Description: "*V. Blanco, I. Espejo, R. Páez, A.M. Rodríguez-Chía, A mathematical optimization approach to Multisphere Support Vector Data Description,* [arxiv:2507.11106](https://arxiv.org/abs/2507.11106)"
- Optimization with [Gurobi](https://www.gurobi.com/).

## 📄 License

This project is under GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for more details. The use of Gurobi requires a separate license. The proyect is distributed as is, without any warranty neither expressed nor implied of their functionality, suitability, or usability.

## ✍️ Authors

- **Víctor Blanco**
- **Inmaculada Espejo**
- **Raúl Páez**
- **Antonio M. Rodríguez-Chía**

## 📧 Support

To report bugs or suggest improvements, open an issue in the repository.

---

**Last update**: December 2025
