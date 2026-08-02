# PHENO_MaizE Pipeline

An open-source modular pipeline for UAV-based high-throughput phenotyping in maize using vegetation indices extracted from UAV orthomosaics.

The pipeline provides a reproducible workflow for vegetation-index extraction, missing-value imputation, machine-learning prediction, validation, and genotype-selection analyses.

The software supports both:

- a unified command-line workflow through **`run_pipeline.py`**
- standalone execution of individual analysis modules for advanced users

Developed within the **PHENO_MaizE** project (Grant No 6672, Science Fund of the Republic of Serbia).

> 🚀 **Status:** Stable release: **v1.1.0**
---

## Overview

PHENO_MaizE Pipeline is an open-source software package for reproducible UAV-based high-throughput phenotyping in maize.

The pipeline extracts vegetation indices from UAV orthomosaics, imputes missing feature values, and provides a unified framework for predicting agronomic traits using multiple machine-learning algorithms and validation strategies.

The software can be used in two complementary ways:

- **Unified workflow** through `run_pipeline.py`, which orchestrates the complete analysis pipeline from preprocessing to model evaluation.
- **Standalone modules**, allowing individual scripts to be executed independently for customized workflows or integration into existing analysis pipelines.

The prediction framework is fully configurable through command-line options, enabling systematic comparison of:

- vegetation indices
- plot mask types
- growth stages
- machine-learning algorithms
- validation strategies

without modifying the source code.

The current version supports prediction of the following traits:

- grain yield
- grain moisture
- days to anthesis (DTA)

---

## Key features

- Unified command-line interface through `run_pipeline.py` for executing complete or partial workflows.
- Modular architecture allowing standalone execution of all pipeline components.
- End-to-end workflow from UAV orthomosaics to agronomic trait prediction.
- Flexible preprocessing including vegetation-index extraction and missing-value imputation.
- Configurable feature selection based on vegetation indices, plot mask types, growth stages, and feature statistics.
- Support for multiple machine-learning algorithms (Lasso, Ridge, Random Forest, XGBoost, CNN).
- Multiple validation strategies, including repeated cross-validation, GroupKFold by genotype, and reciprocal location validation.
- Genotype selection-efficiency analysis based on GroupKFold predictions.
- Reproducible analyses through a consistent command-line interface.

---

## Workflow

The recommended way to use PHENO_MaizE Pipeline is through the unified command-line interface provided by `run_pipeline.py`.

```text
                           run_pipeline.py
                                  │
          ┌───────────────────────┼────────────────────────┐
          │                       │                        │
          ▼                       ▼                        ▼
   preprocess               analysis                  full workflow
          │                       │
          ▼                       ▼
   ┌──────────────┐      ┌──────────────────────────────────────────┐
   │ extract.py   │      │ ml.py                                    │
   └──────┬───────┘      │ group_kfold_predictions.py               │
          ▼              │ location_validation.py                   │
   ┌──────────────┐      │ selection_efficiency.py                  │
   │ impute.py    │      └──────────────────────────────────────────┘
   └──────────────┘
```

The pipeline can be executed at different levels depending on the desired workflow:

- **`preprocess`** – performs vegetation-index extraction followed by missing-value imputation.
- **`analysis`** – executes one or more prediction and validation modules using an imputed dataset.
- **`full`** – runs the complete workflow from UAV-derived vegetation-index extraction to agronomic trait prediction and evaluation.

For advanced users, every module can also be executed independently.

---

## Pipeline Components

PHENO_MaizE Pipeline consists of a unified workflow interface together with modular analysis scripts that can be executed independently.

### Unified workflow

| Component | Description |
|-----------|-------------|
| `run_pipeline.py` | Main command-line interface that orchestrates complete or partial PHENO_MaizE workflows. Supports preprocessing, prediction, validation, and full end-to-end analyses through a single entry point. |

---

### Preprocessing modules

| Script | Description |
|--------|-------------|
| `extract.py` | Extracts vegetation indices and canopy cover from UAV orthomosaics and generates long- and wide-format feature tables. |
| `impute.py` | Performs missing-value imputation using MICE or KNN and generates imputed datasets for downstream analyses. |

---

### Prediction and validation modules

| Script | Description |
|--------|-------------|
| `ml.py` | Standard machine-learning prediction using repeated random train/test splits. |
| `group_kfold_predictions.py` | Machine-learning prediction using GroupKFold cross-validation to evaluate prediction of previously unseen genotypes. |
| `location_validation.py` | Reciprocal location validation by training models on one location and testing on another. |
| `selection_efficiency.py` | Evaluates genotype-selection efficiency based on GroupKFold prediction results. |

---

## Requirements

The pipeline has been tested on:

- Windows 11
- Ubuntu 24.04 LTS
- Python 3.11

The software has been tested using Conda environments on both operating systems.

All required dependencies are provided in the accompanying environment.yml file.

After installing all dependencies, the complete workflow can be executed through the unified command-line interface (`run_pipeline.py`) or by running individual modules independently.

---

## Getting Started

Clone the repository:

```bash
git clone https://github.com/sofijabozinovic/phenomaize-pipeline.git
cd phenomaize-pipeline
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate phenomaize
```

Verify that the installation was successful:

```bash
python src/run_pipeline.py --help
```

This command displays all available workflow modules and analysis options.

The complete example dataset is available on Zenodo:

**https://doi.org/10.5281/zenodo.21676750**

The dataset includes both the raw files required to reproduce the complete workflow (including UAV orthomosaics) and processed example tables for testing the prediction modules.

Download and extract the dataset before running the examples below preserving the original directory structure.

---

## Quick Start

The recommended way to use PHENO_MaizE Pipeline is through the unified command-line interface provided by `run_pipeline.py`.

### Run the complete workflow

```bash
python src/run_pipeline.py full --target yield
```

This command automatically performs:

1. vegetation-index extraction
2. missing-value imputation
3. machine-learning prediction
4. GroupKFold validation
5. reciprocal location validation
6. selection-efficiency analysis

### Run preprocessing only

```bash
python src/run_pipeline.py preprocess
```

### Run prediction and validation modules

```bash
python src/run_pipeline.py analysis --target yield
```

### Run a single analysis module

```bash
python src/run_pipeline.py analysis \
    --target yield \
    --analyses group-kfold
```

### Preview workflow commands

```bash
python src/run_pipeline.py full --target yield --dry-run
```

### Re-run completed steps

```bash
python src/run_pipeline.py full --target yield --force
```

## Advanced Usage: Standalone Modules

The unified workflow (`run_pipeline.py`) is recommended for most users.

The standalone scripts described below remain available for customized workflows, development, and advanced analyses.

### Standard machine-learning evaluation

```bash
python src/ml.py \
    --input path/to/dataset/results/vi_results_wide_imputed_mice_global.csv \
    --target yield
```

### GroupKFold validation

```bash
python src/group_kfold_predictions.py \
    --input path/to/dataset/results/vi_results_wide_imputed_mice_global.csv \
    --target yield \
    --group-col genotype
```

### Selection-efficiency analysis

```bash
python src/selection_efficiency.py \
    --predictions path/to/dataset/results/groupkfold_yield/predictions.csv 
```

### Reciprocal location validation

```bash
python src/location_validation.py \
    --input path/to/dataset/results/vi_results_wide_imputed_mice_global.csv \
    --target yield
```
---

## Customizing Analyses

All prediction modules share a common command-line interface, allowing analyses to be customized without modifying the source code.

The unified workflow (`run_pipeline.py`) forwards these options to the corresponding analysis modules, providing the same flexibility through a single entry point.

Common configuration options include:

- machine-learning algorithm
- vegetation indices
- plot mask type
- growth stages
- vegetation-index statistics
- canopy cover inclusion

These options can be freely combined, allowing systematic comparison of different feature subsets and analysis settings.

Most prediction modules also support custom feature filtering using regular expressions.

To display all available options for a specific module, use:

```bash
python src/ml.py --help
```

or replace `ml.py` with any other standalone analysis module.

To display all available options for the unified workflow, use:

```bash
python src/run_pipeline.py analysis --help
```
---

## Example Dataset

An example dataset is publicly available on Zenodo:

**https://doi.org/10.5281/zenodo.21676750**

The dataset enables users to reproduce the complete PHENO_MaizE workflow and explore the PHENO_MaizE Pipeline without requiring access to proprietary breeding data.

The archive contains:

- UAV orthomosaics for all flights
- flight metadata
- field metadata
- processed vegetation-index tables
- example datasets for machine-learning analyses

The example dataset can be used with:

- `run_pipeline.py` for executing the complete workflow
- individual standalone modules for customized analyses

The dataset is intended for demonstration, testing, and reproducibility of the PHENO_MaizE Pipeline.

---

## Output Files

Depending on the selected workflow, PHENO_MaizE Pipeline automatically organizes results into dedicated output directories.

Typical outputs include:

- extracted vegetation-index tables
- imputed datasets
- trained machine-learning models
- prediction summaries
- prediction tables
- feature-importance tables
- reciprocal location-validation results
- genotype selection-efficiency analyses
- automatically generated reports for each analysis module

The default output structure is:

```text
results/
├── extracted/
├── imputed/
├── ml/
├── group_kfold/
├── location_validation/
└── selection_efficiency/
```

Custom output locations can be specified through the command-line interface.

---

## Reproducibility

All example commands provided in this README have been tested using the complete example dataset together with the accompanying `environment.yml` file on both Windows and Ubuntu.

The unified workflow and all standalone analysis modules produce reproducible results when executed with the same input data and software environment.

---

## Acknowledgements

This software was developed within the **PHENO_MaizE** project.

The project is supported by the **Science Fund of the Republic of Serbia** (Grant No. **6672**), *High-throughput field phenotyping in temperate maize hybrid breeding: how can phenomics improve speed and accuracy of selection?*

---

## Citation

If you use PHENO_MaizE Pipeline in your research, please cite both the software and, when applicable, the accompanying example dataset.

**Software (v1.1.0)**

https://doi.org/10.5281/zenodo.21739659

**Example dataset**

https://doi.org/10.5281/zenodo.21676750

Citation metadata are provided through the `CITATION.cff` file and GitHub's **Cite this repository** feature.

---

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.

---

PHENO_MaizE Pipeline is actively developed within the PHENO_MaizE project. Contributions, suggestions, and bug reports are welcome through GitHub Issues.

