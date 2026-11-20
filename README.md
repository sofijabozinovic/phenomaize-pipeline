# PHENO_MaizE Pipeline

An open-source modular pipeline for UAV-based high-throughput phenotyping in maize using RGB-derived vegetation indices from UAV orthomosaics.

The pipeline provides a reproducible workflow for vegetation-index extraction, missing-value imputation, machine-learning prediction, validation, and genotype-selection analyses.

The software supports both:

- a unified command-line workflow through **`run_pipeline.py`**
- standalone execution of individual analysis modules for advanced users

Developed within the **PHENO_MaizE** project (Grant No 6672, Science Fund of the Republic of Serbia).

> **Release documentation:** v1.3.0 (existing published release). The updated pipeline described here is being validated for a future release; version and DOI will be confirmed before publication.

---

## Overview

PHENO_MaizE Pipeline is an open-source software package for reproducible UAV-based high-throughput phenotyping in maize.

The pipeline extracts RGB-derived vegetation indices from UAV orthomosaics, imputes missing feature values, and provides a unified framework for predicting agronomic traits using multiple machine-learning algorithms and validation strategies.

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
- Phase-level prediction experiments combining acquisitions from the same phenological phase while retaining acquisition-specific features as separate predictors.
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
- **`analysis`** – executes one or more prediction and validation modules using a feature table; for unbiased evaluation with fold-wise imputation, use a table that has not been globally imputed.
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

A clean Conda installation and an analytical workflow have been successfully tested on a Windows computer without an NVIDIA GPU. The installation test verified imports of the principal dependencies and execution of the unified CLI. The analytical test used Ridge, the original wide feature table, and fold-wise MICE imputation across ML prediction, GroupKFold, reciprocal location validation, and selection-efficiency analysis.

The environment allows Python 3.11 or 3.12. Confirmed testing on Ubuntu 24.04 LTS, the exact Windows edition, and additional Python versions should be documented separately when verified.

The accompanying `environment.yml` specifies the main Conda and pip dependencies. Package versions are not fully pinned; for exact reproduction, preserve an export of the tested resolved environment. The unpinned dependencies may resolve to different versions on later installations.

After installing all dependencies, the workflow can be executed through the unified command-line interface (`run_pipeline.py`) or by running individual modules independently. Successful clean-install testing of the analytical workflow does not, by itself, verify fresh orthomosaic extraction or the entire end-to-end workflow.

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

### CPU and optional NVIDIA GPU support

The standard `environment.yml` installs PyTorch through pip, but does not guarantee a CUDA-enabled build. On a computer without a supported NVIDIA GPU, PyTorch can run the CNN on the CPU. For NVIDIA GPU acceleration, install a compatible CUDA-enabled PyTorch build following the official PyTorch installation instructions for the target operating system and driver, then verify `torch.cuda.is_available()`. GPU availability alone does not prove that every script uses the GPU; verify device selection during an actual CNN run.

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The documented clean-install test used a CPU-only PyTorch build. NVIDIA GPU execution remains to be checked separately.

The complete example dataset is available on Zenodo:

**https://doi.org/10.5281/zenodo.21676750**

The example dataset includes four representative UAV RGB orthomosaics with corresponding input files, along with extracted and imputed phenotypic features derived from these four orthomosaics for testing the pipeline modules.

Download and extract the dataset before running the examples below, preserving the original directory structure.

---

### Important: Plot identifier consistency

For successful vegetation-index extraction and integration with field data, plot identifiers must be consistent across input files.

- The `PlotID` attribute in each `shapefiles/<trial>/plots.shp` file must exactly match the corresponding `plot_id` values in `field_data.csv`.
- The `trial` values in `flight_info.csv` and `field_data.csv` must match exactly and correspond to the names of the trial-specific folders within `shapefiles/`.
- Plot identifiers must be unique within each trial.

For example, if a plot is identified as `SD-TOH-P49` in `field_data.csv`, the corresponding `PlotID` in the shapefile must also be `SD-TOH-P49`, not simply `P49`.

**Important:** Mismatched plot identifiers may result in missing vegetation-index values in the final wide-format output (`vi_results_wide.csv`), even when extraction has completed successfully. Users should verify identifier consistency before running the pipeline.

## Quick Start

The recommended way to use PHENO_MaizE Pipeline is through the unified command-line interface provided by `run_pipeline.py`.

The following commands assume that the downloaded example dataset has been organized in the `example_data/` directory. Use `--data-dir` to specify a different location.

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

**Note:** The example dataset contains four orthomosaics and corresponding extracted and imputed feature tables, enabling users to test both preprocessing and downstream analysis workflows.

### Run preprocessing only

```bash
python src/run_pipeline.py preprocess
```

### Run prediction and validation modules

```bash
python src/run_pipeline.py analysis \
    --input path/to/dataset/results/vi_results_wide_imputed_mice_global.csv \
    --target yield
```

### Run a single analysis module

```bash
python src/run_pipeline.py analysis \
    --input path/to/dataset/results/vi_results_wide_imputed_mice_global.csv \
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
    --predictions results/group_kfold/yield/veg_qgis_mean_std/predictions.csv
```

### Reciprocal location validation

```bash
python src/location_validation.py \
    --input path/to/dataset/results/vi_results_wide_imputed_mice_global.csv \
    --target yield
```
**Note:** Replace `path/to/dataset/results/` with the actual directory containing the downloaded preprocessed CSV files.

---

## Customizing Analyses

Prediction modules expose overlapping command-line options, allowing analyses to be customized without modifying the source code.

The unified workflow (`run_pipeline.py`) forwards these options to the corresponding analysis modules, providing the same flexibility through a single entry point.

Common configuration options include:

- machine-learning algorithm
- vegetation indices
- plot mask type
- growth stages
- vegetation-index statistics
- canopy cover inclusion

These options can be combined where supported by the relevant module, allowing systematic comparison of different feature subsets and analysis settings.

**Default feature selection.** Unless otherwise specified, the prediction modules use vegetation-index features extracted using the QGIS vegetation mask (`veg_qgis`), including both mean and standard deviation statistics (`mean_std`). Canopy cover features are excluded by default and can be enabled using `--include-canopy`. Users can override these settings through `--feature-set` and `--stats` to select alternative vegetation masks and feature statistics.

### Missing-data handling during model validation

The standalone `impute.py` module can generate globally imputed feature tables for exploratory or preprocessing workflows. For evaluation of predictive performance, however, globally imputing the full dataset before splitting it may leak information from validation observations into training. The updated prediction modules support training-only missing-data handling (`--missing-policy fold_mice`, `fold_knn`, or `drop`), fitting the imputer on the training subset within each split or fold and applying that fitted transformation to held-out data. Reciprocal location validation fits preprocessing using the training location only. Prefer supplying a feature table that has **not** been globally imputed when using fold-wise imputation.

For example, the following analysis command uses the original wide feature table and fold-wise MICE:

```bash
python src/run_pipeline.py analysis \
    --input path/to/dataset/results/vi_results_wide.csv \
    --target yield \
    --feature-set veg_qgis \
    --stats mean \
    --include-canopy \
    --missing-policy fold_mice
```

### Example: Custom feature selection

The following example runs all five machine-learning models and the available prediction, validation, and selection-efficiency analyses using a wide feature table with training-only missing-data handling. It selects vegetation indices extracted using the QGIS vegetation mask, includes both mean and standard deviation statistics, and enables canopy cover features.

```bash
python src/run_pipeline.py analysis \
    --input path/to/dataset/results/vi_results_wide.csv \
    --target yield \
    --feature-set veg_qgis \
    --stats mean_std \
    --include-canopy \
    --models Lasso Ridge RF XGBoost CNN \
    --missing-policy fold_mice \
    --show-feature-summary
```

Replace `path/to/dataset/results/` with the actual directory containing the preprocessed CSV file. The `--show-feature-summary` option prints a summary of the selected predictors, allowing users to verify the feature configuration before interpreting the results.

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

The example dataset provides raw input files and preprocessed data for testing the PHENO_MaizE Pipeline.

The archive contains:
- Four representative UAV RGB orthomosaics (two per location).
- Corresponding plot boundary shapefiles, field data, and flight metadata.
- Extracted vegetation indices and canopy cover from the four provided orthomosaics.
- Preprocessed datasets for machine-learning analyses.

The provided orthomosaics can be used to test feature extraction, while the precomputed datasets enable prediction, validation, and genotype selection-efficiency analyses without processing the complete image collection.

The example dataset can be used with:

- `run_pipeline.py` for testing preprocessing on the provided orthomosaics and running prediction and validation analyses using the precomputed datasets.
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

Prediction and validation experiments are organized by broad phenological phase (`Emerg`, `Veg`, `Flow`, `Grain`, `Mat`). All acquisition dates belonging to the same phase are included in a single phase-specific experiment, while the features from each acquisition remain separate predictors rather than being averaged across dates. A combined `All_Phases` experiment is also generated (`All_Selected_Stages` when specific stages are selected). GroupKFold additionally organizes model-specific outputs by cross-validation fold. Selection-efficiency analysis uses GroupKFold predictions to evaluate genotype selection across experiments and models.

Prediction and validation results are stored under the analysis module, target trait, and a feature-configuration directory named `<feature_set>_<stats>`, with `_canopy` appended when canopy cover is included. The GroupKFold **script** is `group_kfold_predictions.py`, while its **results directory** is `group_kfold/`. An illustrative output structure is:

```text
results/
├── extracted/
├── imputed/
├── ml/
│   └── yield/
│       └── veg_qgis_mean_std_canopy/
│           ├── Emerg/
│           ├── Veg/
│           ├── Flow/
│           ├── Grain/
│           ├── Mat/
│           ├── All_Phases/
│           ├── results_summary.csv
│           └── report.md
├── group_kfold/
│   └── yield/
│       └── veg_qgis_mean_std_canopy/
│           ├── Emerg/
│           ├── All_Phases/
│           ├── predictions.csv
│           └── results_summary_mean.csv
├── location_validation/
│   └── yield/
│       └── veg_qgis_mean_std_canopy/
│           ├── predictions.csv
│           └── location_validation_report.md
└── selection_efficiency/
    └── yield/
        └── <configuration_id>/
            ├── selection_summary.csv
            └── selection_report.md
```

The directory tree is illustrative: experiment folders and output files depend on the selected modules, features, and settings. GroupKFold experiment folders contain fold-specific results.

Custom output locations can be specified through the command-line interface. **Note:** Runs that share the same target, feature set, statistics, and canopy setting may write to the same directory even if other options (e.g., vegetation-index filters or selected stages) differ. Use separate `--results-dir` locations to preserve distinct configurations; `--force` overwrites/recomputes existing results.

---

## Reproducibility

Analytical workflows have been tested using the provided feature data, including a successful clean Conda installation and CPU-based Ridge analysis. Two repeated CNN runs with the same settings produced identical reported evaluation metrics in the tested environment. This is evidence of within-environment metric reproducibility, not a guarantee of identical results across operating systems, PyTorch builds, or CPU/GPU hardware.

Reproducibility depends on the input data, software versions, random seeds, and computational environment. For an exact computational record, archive the resolved environment (for example, `conda list --explicit` and `pip freeze`) alongside the analysis configuration and outputs.

---

## Acknowledgements

This software was developed within the **PHENO_MaizE** project.

The project is supported by the **Science Fund of the Republic of Serbia** (Grant No. **6672**), *High-throughput field phenotyping in temperate maize hybrid breeding: how can phenomics improve speed and accuracy of selection?*

---

## Citation

If you use PHENO_MaizE Pipeline in your research, please cite both the software and, when applicable, the accompanying example dataset.

**Existing software release v1.3.0**

https://doi.org/10.5281/zenodo.24275693

**Example dataset**

https://doi.org/10.5281/zenodo.21676750

Citation metadata are provided through the `CITATION.cff` file and GitHub's **Cite this repository** feature. Update the release number, citation metadata, and version-specific Zenodo DOI when publishing the next release.

---

## License

This project is distributed under the MIT License. See the `LICENSE` file for details.

---

PHENO_MaizE Pipeline is actively developed within the PHENO_MaizE project. Contributions, suggestions, and bug reports are welcome through GitHub Issues.

