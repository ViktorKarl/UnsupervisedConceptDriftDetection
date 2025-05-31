# Unsupervised Concept Drift Detection

This repository extends the original work by Lukats et al. (https://github.com/DFKI-NI/unsupervised-concept-drift-detection), introducing new detectors, enhanced hyperparameter optimization, improved workflows, and reproducible experiments for fully unsupervised concept drift detection on real-world and synthetic data streams.

The paper by Lukats et al. is published in the *International Journal of Data Science and Analytics* (https://link.springer.com/article/10.1007/s41060-024-00620-y).

---

## Table of Contents

1. [Branches](#branches)  
2. [Overview](#overview)  
3. [Implemented Detectors](#implemented-detectors)  
4. [Hyperparameter Optimization](#hyperparameter-optimization)  
5. [Setup Guide](#setup-guide)  
6. [Execution Guide](#execution-guide)  

---

## Branches

| Branch                 | Purpose                                                                                                                    |
|------------------------|----------------------------------------------------------------------------------------------------------------------------|
| `main`                 | Contains the **extended framework** with all novel contributions: three additional distance-based detectors, Optuna-powered hyperparameter search, and improved logging. |
| `reproduction_results` | Contains results from reproducing the experiments reported by **Lukats et al.**                                           |

---

## Overview

Concept drift detectors identify changes in the statistical properties of evolving data streams that may silently degrade a model’s predictive performance. This repository provides:

- **Six fully unsupervised** concept drift detectors (three from the original study and three new distance-based variants).  
- A **configurable workflow** for large-scale benchmarking on real-world (**USP DS Repository**) and synthetic data streams.  
- **Optuna** integration for black-box hyperparameter optimization, including a web dashboard.  
- A **cut-off method** to reduce false positives during the detection of prolonged drift.  
- **Adjustable step-size** for window sliding, optimized via hyperparameter search to potentially reduce computation.  

---

## Implemented Detectors

| Detector                                 | Abbreviation | Reference                                                                        |
|------------------------------------------|--------------|----------------------------------------------------------------------------------|
| Bayesian Non-parametric Detection Method | **BNDM**     | Fang et al., *ACM TIST (2021)* [[link]](https://doi.org/10.1145/3420034)         |
| Discriminative Drift Detector            | **D3**       | Sethi & Tudoreanu, *KDD ’19* [[link]](https://doi.org/10.1145/3357384.3358144)   |
| Semi-Parametric Log-Likelihood           | **SPLL**     | Ross et al., *IEEE TKDE (2012)* [[link]](https://doi.org/10.1109/TKDE.2011.226)  |
| Jensen–Shannon Distance Detector         | **JSD**      | *This work*                                                                      |
| Kullback–Leibler Divergence Detector     | **KLD**      | *This work*                                                                      |
| Hellinger Distance Detector              | **HD**       | *This work*                                                                      |

➡️ The three new detectors compute a sliding-window divergence score between distributions and signal drift when the score exceeds an adaptive threshold.

Additionally, some detectors implemented by Lukats et al. are included in the `detectors` folder but are not currently compatible with the framework. Future work may integrate them into the system.

---

## Hyperparameter Optimization

This repository includes an enhanced hyperparameter optimization (HPO) framework using Optuna, which automatically identifies optimal hyperparameter combinations. Optuna evaluates various configurations, intelligently avoiding ranges that consistently yield poor results based on prior trials. Hyperparameter search spaces for each detector are defined in `config.py`.

---

## Setup Guide

### 1. Clone the Repository

```sh
git clone https://github.com/ViktorKarl/UnsupervisedConceptDriftDetection.git
cd UnsupervisedConceptDriftDetection
```

### 2. Create a Virtual Environment (Recommended)

This project requires Python 3.8. Create and activate a virtual environment:

```sh
python3.8 -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

### 3. Install Dependencies

Install the required packages:

```sh
pip install -r requirements.txt
```

### 4. Download and Prepare Datasets

- Download datasets from the USP DS Repository (https://sites.google.com/view/uspdsrepository) and extract them to `datasets/files/`. Note that dataset column structures may differ from those used in this work.

---

## Execution Guide

### 1. Configure Your Experiment

Edit [`config.py`](config.py) to specify:

```sh
model_selection: True or False          # Select any subset of models
stream_selection: True or False         # Or use custom CSVs in datasets/files/
evaluation_metric: "1_metric"           # Implemented in evaluation/
study_name: "my_study"                  # Set your desired study name
optuna_settings:
    n_trials: 50                        # Number of Optuna trials per detector
    storage: "sqlite:///lite_optuna.db" # Storage database to be used by Optuna. If the provided path points to a non-existent database, it will be created from scratch
cut_off: False                          # Removed overshooting drift points for an extended drift
benchmark_metrics: True                 # Enable logging of LPD and Accuracy using Hoeffding tree
```

### 2. Run an Experiment

Start an experiment with:

```sh
python main.py
```

Results are stored in the database specified in [`config.py`](config.py). All trials can be accessed via the Optuna dashboard.

### 3. Visualize Optimization (Optional)

Use the Optuna Dashboard to inspect optimization studies:

```sh
optuna-dashboard sqlite:///lite_optuna.db
```

Navigate to `http://localhost:8080/` to monitor trial statistics, parameter importance, and parallel coordinate plots.

---

Happy drifting! 🚀
