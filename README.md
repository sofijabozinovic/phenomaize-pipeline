# PHENO_MaizE Pipeline

Machine-learning framework for UAV-based high-throughput phenotyping in maize using vegetation indices extracted from UAV orthomosaics.

Designed for reproducible extraction, prediction, and evaluation of agronomic traits from UAV imagery.

Developed within the **PHENO_MaizE** project.

> 🚧 **Status:** Under active development.

---

## Overview

PhenoMaize Pipeline is an open-source machine-learning framework for UAV-based high-throughput phenotyping in maize.

The framework extracts vegetation indices from UAV orthomosaics and provides a reproducible workflow for evaluating their ability to predict agronomic traits.

The prediction modules are fully configurable through command-line options, allowing users to compare different:

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

- End-to-end workflow from UAV orthomosaics to agronomic trait prediction.
- Flexible command-line interface for reproducible analyses.
- Configurable selection of vegetation indices, plot mask types, growth stages, and feature statistics.
- Support for multiple machine-learning algorithms (Lasso, Ridge, Random Forest, XGBoost, CNN).
- Multiple validation strategies, including repeated cross-validation, GroupKFold by genotype, and reciprocal location validation.
- Genotype selection-efficiency analysis based on GroupKFold predictions.

---

## Workflow

```text
extract.py
      ↓
impute.py
      ↓
prediction modules
      ↓
analysis outputs
```

After feature extraction and imputation, the resulting dataset can be analyzed using one or more prediction modules:

- standard machine-learning evaluation (`ml.py`)
- GroupKFold validation (`group_kfold_predictions.py`)
- reciprocal location validation (`location_validation.py`)
- genotype selection-efficiency analysis (`selection_efficiency.py`, based on GroupKFold predictions)

---

## Pipeline components

| Script | Description |
|----------|-------------|
| `extract.py` | Extracts plot-level vegetation indices from UAV orthomosaics using plot polygons. |
| `impute.py` | Imputes missing feature values using multiple statistical methods. |
| `ml.py` | Trains and compares machine-learning models using configurable feature subsets. |
| `group_kfold_predictions.py` | Evaluates prediction performance using GroupKFold validation while keeping complete genotype groups together. |
| `selection_efficiency.py` | Quantifies genotype-selection efficiency from GroupKFold predictions. |
| `location_validation.py` | Evaluates model transferability between experimental locations. |

---

## Requirements

The pipeline has been tested on:

- Windows 11
- Ubuntu 24.04 LTS
- Python 3.11

The software has been tested using Conda environments on both operating systems.

All required dependencies are provided in the accompanying environment.yml file.

---

## Getting started

Clone the repository:

```bash
git clone https://github.com/sofijabozinovic/phenomaize-pipeline.git
cd phenomaize-pipeline
```

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate phenomaize

The repository includes an example dataset that allows users to reproduce the complete workflow.

Run the preprocessing steps:

```bash
python extract.py
python impute.py
```

The resulting imputed dataset can then be used with one or more prediction modules, depending on the desired analysis.

---

## Running prediction modules

After preprocessing, the imputed dataset can be analyzed using one or more prediction modules. 
The following examples illustrate the most common analyses.

### Standard machine-learning evaluation

```bash
python ml.py \
    --input example_data/results/vi_results_wide_imputed_mice_global.csv \
    --target yield
```

### GroupKFold validation

```bash
python group_kfold_predictions.py \
    --input example_data/results/vi_results_wide_imputed_mice_global.csv \
    --target yield \
    --group-col genotype
```

### Selection-efficiency analysis

```bash
python selection_efficiency.py \
    --predictions example_data/results/groupkfold_yield/predictions.csv
```

### Reciprocal location validation

```bash
python location_validation.py \
    --input example_data/results/vi_results_wide_imputed_mice_global.csv \
    --target yield
```
---

## Customizing analyses

The prediction scripts share a common command-line interface, allowing analyses to be customized without modifying the source code.

Common configuration options include:

- machine-learning model
- vegetation indices
- plot mask type
- growth stages
- vegetation-index statistics
- canopy cover inclusion

These options can be combined, allowing users to systematically compare different feature subsets and analysis settings.

Most prediction scripts also support custom feature filtering using regular expressions.

To display all available options for a script, use:

```bash
python ml.py --help
```

or replace `ml.py` with the desired prediction script.

---

## Example dataset

The repository includes a small example dataset that allows users to test the complete pipeline without preparing their own data.

The example dataset includes:

- UAV orthomosaics
- experimental plot polygons
- field measurements
- flight metadata

All scripts in this repository have been tested using this dataset on both Windows and Ubuntu.

---

## Output files

Depending on the selected workflow, the pipeline generates:

- extracted vegetation indices
- imputed datasets
- trained machine-learning models
- prediction tables
- feature importance tables
- genotype selection-efficiency summaries
- reciprocal location-validation results

All output files are written to user-specified output directories.

---

## Reproducibility

All example commands provided in this README have been tested using the example dataset and the accompanying `environment.yml` file on both Windows and Ubuntu.

---

## Acknowledgements

This software was developed within the **PHENO_MaizE** project.

The project is supported by the **Science Fund of the Republic of Serbia** (Grant No. **6672**), *High-throughput field phenotyping in temperate maize hybrid breeding: how can phenomics improve speed and accuracy of selection?*

---

## Citation

If you use this software in your research, please cite it using the metadata provided in the `CITATION.cff` file or the GitHub **Cite this repository** option.

---

## License

License information will be added after institutional approval.


