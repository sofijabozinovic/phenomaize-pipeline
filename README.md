# PhenoMaize Pipeline

Machine learning pipeline for UAV-based high-throughput phenotyping in maize using UAV-derived vegetation indices.
Developed within the PHENO_MaizE project.

> 🚧 **Status:** Under active development.

---

## Overview

PhenoMaize Pipeline is an open-source workflow developed within the **PHENO_MaizE** project for predicting agronomic traits from UAV imagery.

The pipeline integrates:

- vegetation index extraction
- missing value imputation
- machine learning prediction
- genotype selection efficiency analysis
- reciprocal location validation

The current version supports prediction of:

- Grain yield
- Grain moisture
- Days to anthesis (DTA)
---

## Workflow

```text
extract.py
      ↓
impute.py
      ↓
group_kfold_predictions.py
      ↓
selection_efficiency.py
      ↓
location_validation.py
```

Each script performs one step of the pipeline and produces outputs required by the next stage, enabling a reproducible end-to-end workflow.
---

## Pipeline components

| Script | Description |
|----------|-------------|
| `extract.py` | Extracts vegetation indices from UAV orthomosaics for each experimental plot. |
| `impute.py` | Handles missing values using several imputation methods. |
| `group_kfold_predictions.py` | Performs GroupKFold cross-validation for robust prediction of agronomic traits. |
| `selection_efficiency.py` | Evaluates genotype selection efficiency based on predicted trait values. |
| `location_validation.py` | Evaluates model transferability between locations using reciprocal validation. |
---

## Requirements

- Python 3.11 or newer
- Windows (tested)
- Required Python packages (to be listed in `requirements.txt`)

---

## Getting started

Clone the repository:

```bash
git clone https://github.com/sofijabozinovic/phenomaize-pipeline.git
cd phenomaize-pipeline
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Prepare the input data following the example directory structure.

Run the pipeline step by step:

```bash
python extract.py
python impute.py
python group_kfold_predictions.py
python selection_efficiency.py
python location_validation.py
```
