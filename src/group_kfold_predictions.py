#!/usr/bin/env python3
"""
PhenoMaize GroupKFold prediction module

OUTPUT:
- <output_dir>/<Experiment>/<Split>/ plots + feature importance where applicable
- <output_dir>/performance_comparison.png
- <output_dir>/report.md
- <output_dir>/results_summary.csv
- <output_dir>/results_summary_mean.csv

Supports:
- feature-set: all / veg_qgis / veg_exg80 / combined
- stats: mean / std / mean_std
- optional canopy cover features
- include/exclude regex filters
- target leakage protection
- models: Lasso, Ridge, RF, XGBoost, CNN (CNN trains+evaluates but NO feature importance)
- experiments: automatically inferred from feature column prefixes (generalized)
- cv-mode: random / hybrid_groupkfold
"""

import os
import sys
import argparse
import re
import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

warnings.filterwarnings("ignore")


# -----------------------------
# Configuration
# -----------------------------
BLACKLIST = {
    "location",
    "material_type",
    "plot_id",
    "fao",
    "rep",
    "genotype",
    "trial",
}

# Known response variables excluded from predictors to reduce target leakage.
# --target itself is flexible and may be any numeric column in the input CSV.
TARGET_COLUMNS = {"yield", "moisture", "dta"}


# -----------------------------
# CNN model
# -----------------------------
class PyTorchCNN(nn.Module):
    """Simple MLP-style network for regression on tabular data."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------
# Plot helpers
# -----------------------------
def plot_feature_importance(df_imp, out_path, title):
    df_imp = df_imp.sort_values("Importance", ascending=False).head(15)
    plt.figure(figsize=(10, 8))
    sns.barplot(data=df_imp, x="Importance", y="Feature", palette="viridis")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_actual_vs_pred(y_true, y_pred, out_path, title):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6)

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))
    plt.plot([mn, mx], [mn, mx], "--r", linewidth=2)

    plt.title(title)
    plt.xlabel("Actual values")
    plt.ylabel("Predicted values")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# Feature selection
# -----------------------------
def _base_numeric_features(df, target):
    """Return numeric candidate predictors while excluding metadata and targets."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    blacklist_lower = {str(c).lower() for c in BLACKLIST}
    target_columns_lower = {str(c).lower() for c in TARGET_COLUMNS}
    target_lower = str(target).lower()

    feats = [
        c for c in num_cols
        if str(c).lower() not in blacklist_lower
        and str(c).lower() not in target_columns_lower
        and str(c).lower() != target_lower
    ]

    leaked = [c for c in feats if str(c).lower() in target_columns_lower]
    if leaked:
        raise RuntimeError(
            f"Leakage: target columns present in features: {sorted(leaked)}"
        )
    return feats


def select_features(df, target, feature_set="veg_qgis", stats="mean_std",
                    include_canopy=False, include_regex=None, exclude_regex=None,
                    show_summary=False):
    """Select predictors using the same rules as ml.py."""
    feats = _base_numeric_features(df, target)

    want_mean = stats in ("mean", "mean_std")
    want_std = stats in ("std", "mean_std")
    vi_keep = []

    def add_suffix(suf_mean, suf_std):
        if want_mean:
            vi_keep.extend([c for c in feats if c.endswith(suf_mean)])
        if want_std:
            vi_keep.extend([c for c in feats if c.endswith(suf_std)])

    if feature_set == "all":
        add_suffix("_mean_all", "_std_all")
    elif feature_set == "veg_qgis":
        add_suffix("_mean_veg_qgis", "_std_veg_qgis")
    elif feature_set == "veg_exg80":
        add_suffix("_mean_veg_exg80", "_std_veg_exg80")
    elif feature_set == "combined":
        # combined intentionally means all pixels + QGIS mask.
        add_suffix("_mean_all", "_std_all")
        add_suffix("_mean_veg_qgis", "_std_veg_qgis")
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    vi_keep = list(dict.fromkeys(vi_keep))

    if include_regex:
        try:
            rgx = re.compile(include_regex, flags=re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid --include-regex: {exc}") from exc
        vi_keep = [c for c in vi_keep if rgx.search(c)]

    canopy_keep = []
    if include_canopy:
        if feature_set in ("veg_qgis", "combined"):
            canopy_keep.extend(
                [c for c in feats if "canopy_cover_qgis" in c.lower()]
            )
        if feature_set in ("veg_exg80", "all", "combined"):
            canopy_keep.extend(
                [c for c in feats if "canopy_cover_exg80" in c.lower()]
            )

    keep = list(dict.fromkeys(vi_keep + canopy_keep))

    if exclude_regex:
        try:
            rgx = re.compile(exclude_regex, flags=re.IGNORECASE)
        except re.error as exc:
            raise ValueError(f"Invalid --exclude-regex: {exc}") from exc
        keep = [c for c in keep if not rgx.search(c)]

    if show_summary:
        print("\nFeature selection summary:")
        print(f"  feature_set={feature_set}, stats={stats}, include_canopy={include_canopy}")
        print(f"  include_regex={include_regex}, exclude_regex={exclude_regex}")
        print(f"  total selected: {len(keep)}")
        for col in keep:
            print(f"    - {col}")
        print()

    if not keep:
        raise RuntimeError(
            "No features selected. Check feature_set, stats and regex filters."
        )

    return keep


# -----------------------------
# Experiments (generalized)
# -----------------------------
def _growth_stage_from_feature(col):
    """Extract the full growth-stage prefix from a feature column name."""
    tails = (
        "_mean_all", "_std_all",
        "_mean_veg_qgis", "_std_veg_qgis",
        "_mean_veg_exg80", "_std_veg_exg80",
        "_canopy_cover_qgis", "_canopy_cover_exg80",
    )

    col_str = str(col)
    for tail in tails:
        if col_str.endswith(tail):
            stem = col_str[:-len(tail)]
            if tail.startswith("_canopy_cover"):
                return stem or "Unknown"
            if "_" in stem:
                return stem.rsplit("_", 1)[0]
    return "Unknown"


def _vegetation_index_from_feature(col):
    """Extract the VI name from a feature column; canopy columns return None."""
    tails = (
        "_mean_all", "_std_all",
        "_mean_veg_qgis", "_std_veg_qgis",
        "_mean_veg_exg80", "_std_veg_exg80",
    )
    col_str = str(col)
    for tail in tails:
        if col_str.endswith(tail):
            stem = col_str[:-len(tail)]
            if "_" in stem:
                return stem.rsplit("_", 1)[1]
    return None


def filter_features_by_exact_vis(feature_cols, vegetation_indices):
    """Keep only exact VI names; canopy columns are preserved separately."""
    if not vegetation_indices:
        return list(feature_cols)

    requested = [str(vi).strip() for vi in vegetation_indices if str(vi).strip()]
    requested_lower = {vi.lower() for vi in requested}
    available = sorted({
        vi for vi in (_vegetation_index_from_feature(col) for col in feature_cols)
        if vi is not None
    })
    available_lower = {vi.lower(): vi for vi in available}
    missing = [vi for vi in requested if vi.lower() not in available_lower]
    if missing:
        raise RuntimeError(
            "Unknown vegetation index/indices: " + ", ".join(missing) + ". "
            "Available indices: " + ", ".join(available)
        )

    return [
        col for col in feature_cols
        if _vegetation_index_from_feature(col) is None
        or _vegetation_index_from_feature(col).lower() in requested_lower
    ]


def define_experiments(feature_cols, stages=None):
    """Create one experiment per stage and one combined experiment."""
    experiments = {}
    for col in feature_cols:
        stage = _growth_stage_from_feature(col)
        experiments.setdefault(stage, []).append(col)
    combined_name = "All_Selected_Stages" if stages else "All_Phases"
    experiments[combined_name] = list(feature_cols)
    return experiments


def filter_features_by_exact_stages(feature_cols, stages):
    """Keep only exact full growth-stage names (case-insensitive)."""
    if not stages:
        return list(feature_cols)

    requested = [str(stage).strip() for stage in stages if str(stage).strip()]
    requested_lower = {stage.lower() for stage in requested}
    available = sorted({_growth_stage_from_feature(col) for col in feature_cols})
    available_lower = {stage.lower(): stage for stage in available}
    missing = [stage for stage in requested if stage.lower() not in available_lower]
    if missing:
        raise RuntimeError(
            "Unknown growth stage(s): " + ", ".join(missing) + ". "
            "Available stages: " + ", ".join(available)
        )

    return [
        col for col in feature_cols
        if _growth_stage_from_feature(col).lower() in requested_lower
    ]


# -----------------------------
# Main pipeline
# -----------------------------
class MLPipeline:
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        self.report_path = os.path.join(self.output_dir, "report.md")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading data from {args.input}...")
        self.df = pd.read_csv(args.input)
        if args.target not in self.df.columns:
            raise ValueError(f"Target '{args.target}' not found in input file.")

        self.df[args.target] = pd.to_numeric(self.df[args.target], errors="coerce")
        if self.df[args.target].notna().sum() < 3:
            raise ValueError(
                f"Target '{args.target}' must contain at least 3 numeric values."
            )

        # Drop rows missing the response variable.
        before = len(self.df)
        self.df = self.df.dropna(subset=[args.target]).reset_index(drop=True)
        after = len(self.df)
        if after < before:
            print(f"Warning: dropped {before-after} rows due to missing target values.")

        # feature selection
        self.feature_cols = select_features(
            self.df,
            target=args.target,
            feature_set=args.feature_set,
            stats=args.stats,
            include_canopy=args.include_canopy,
            include_regex=args.include_regex,
            exclude_regex=args.exclude_regex,
            show_summary=args.show_feature_summary,
        )

        self.feature_cols = filter_features_by_exact_vis(
            self.feature_cols, args.vegetation_indices
        )
        self.feature_cols = filter_features_by_exact_stages(
            self.feature_cols, args.stages
        )

        self.selected_stages = sorted({_growth_stage_from_feature(c) for c in self.feature_cols})
        self.selected_vis = sorted({
            vi for vi in (_vegetation_index_from_feature(c) for c in self.feature_cols)
            if vi is not None
        })
        self.experiments = define_experiments(self.feature_cols, stages=args.stages)
        self.results = []
        self.predictions = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Target: {args.target}")
        print(f"Total selected features: {len(self.feature_cols)}")
        print(f"Experiments found: {list(self.experiments.keys())}")
        print(f"CV mode: {args.cv_mode}")
        print(f"Device: {self.device.upper()}")
        print(f"Output dir: {self.output_dir}")

        # init report
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(f"# Machine learning report: predicting '{self.args.target}'\n\n")
            f.write(f"Generated for input file: `{os.path.basename(self.args.input)}`\n\n")
            f.write("Feature selection:\n")
            f.write(f"- feature_set: `{self.args.feature_set}`\n")
            f.write(f"- stats: `{self.args.stats}`\n")
            f.write(f"- include_canopy: `{self.args.include_canopy}`\n")
            f.write(f"- include_regex: `{self.args.include_regex}`\n")
            f.write(f"- exclude_regex: `{self.args.exclude_regex}`\n")
            f.write(f"- stages: `{self.selected_stages}`\n")
            f.write(f"- vegetation_indices: `{self.selected_vis}`\n")
            f.write(f"- cv_mode: `{self.args.cv_mode}`\n")
            if self.args.cv_mode == "hybrid_groupkfold":
                f.write(f"- group_col: `{self.args.group_col}`\n")
                f.write(f"- n_splits: `{self.args.n_splits}`\n")
            f.write("\n")

    def _train_cnn(self, X_train, y_train, X_test):
        model = PyTorchCNN(X_train.shape[1]).to(self.device)

        Xtr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        ytr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        Xte = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.args.cnn_epochs):
            for xb, yb in loader:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()

        model.eval()
        with torch.no_grad():
            y_pred = model(Xte).cpu().numpy().reshape(-1)

        return y_pred

    def _make_splits(self, X, y, exp_name):
        """Create train/test splits according to selected CV mode."""
        splits = []

        if self.args.cv_mode == "random":
            all_idx = np.arange(len(y))
            train_idx, test_idx = train_test_split(
                all_idx, test_size=0.2, random_state=42
            )
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            splits.append((
                "random_split", X_train, X_test, y_train, y_test, test_idx
            ))

        elif self.args.cv_mode == "hybrid_groupkfold":
            if self.args.group_col not in self.df.columns:
                raise ValueError(
                    f"Group column '{self.args.group_col}' not found in dataframe."
                )

            groups = self.df[self.args.group_col].astype(str).values
            n_unique_groups = len(np.unique(groups))
            n_splits = min(self.args.n_splits, n_unique_groups)

            if n_splits < 2:
                raise ValueError("GroupKFold requires at least 2 unique groups.")

            print(f"Using GroupKFold by '{self.args.group_col}' with {n_splits} folds for {exp_name}.")
            print(f"Unique groups: {n_unique_groups}")

            gkf = GroupKFold(n_splits=n_splits)
            for fold_id, (train_idx, test_idx) in enumerate(
                gkf.split(X, y, groups=groups), start=1
            ):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                splits.append((f"fold_{fold_id}", X_train, X_test, y_train, y_test, test_idx))

        else:
            raise ValueError(f"Unknown cv-mode: {self.args.cv_mode}")

        return splits

    def run(self):
        for exp_name, exp_feature_cols in self.experiments.items():
            print(f"\n{'=' * 70}")
            print(f"Experiment: {exp_name}")
            print(f"{'=' * 70}")

            exp_dir = os.path.join(self.output_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            with open(self.report_path, "a", encoding="utf-8") as f:
                f.write(f"## Experiment: {exp_name}\n\n")
                f.write(f"**Features ({len(exp_feature_cols)}):** `{', '.join(exp_feature_cols)}`\n\n")

            X = self.df[exp_feature_cols].values
            y = self.df[self.args.target].values.reshape(-1)

            splits = self._make_splits(X, y, exp_name)

            for split_name, X_train, X_test, y_train, y_test, test_idx in splits:
                print(f"\n--- Split: {split_name} ---")

                split_dir = os.path.join(exp_dir, split_name)
                os.makedirs(split_dir, exist_ok=True)

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                with open(self.report_path, "a", encoding="utf-8") as f:
                    f.write(f"### Split: {split_name}\n\n")

                for model_name in self.args.models:
                    print(f"  Training {model_name}...", end=" ")

                    try:
                        importance = None

                        if model_name == "CNN":
                            y_pred = self._train_cnn(X_train_scaled, y_train, X_test_scaled)
                        else:
                            if model_name == "Lasso":
                                model = Lasso(alpha=self.args.alpha)
                            elif model_name == "Ridge":
                                model = Ridge(alpha=self.args.alpha)
                            elif model_name == "RF":
                                model = RandomForestRegressor(
                                    n_estimators=self.args.n_estimators,
                                    random_state=42,
                                    n_jobs=-1,
                                )
                            elif model_name == "XGBoost":
                                model = xgb.XGBRegressor(
                                    n_estimators=self.args.n_estimators,
                                    random_state=42,
                                    n_jobs=-1,
                                )
                            else:
                                raise ValueError(f"Unknown model: {model_name}")

                            model.fit(X_train_scaled, y_train)
                            y_pred = model.predict(X_test_scaled).reshape(-1)

                            if model_name in ("RF", "XGBoost"):
                                importance = model.feature_importances_
                            elif model_name in ("Lasso", "Ridge"):
                                importance = np.abs(model.coef_)

                        r2 = float(r2_score(y_test, y_pred))
                        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                        mae = float(mean_absolute_error(y_test, y_pred))

                        self.results.append({
                            "Experiment": exp_name,
                            "Split": split_name,
                            "Model": model_name,
                            "R2": r2,
                            "RMSE": rmse,
                            "MAE": mae,
                            "Target": self.args.target,
                            "Feature_set": self.args.feature_set,
                            "Stats": self.args.stats,
                            "Include_canopy": self.args.include_canopy,
                            "Selected_stages": ";".join(self.selected_stages),
                            "Vegetation_indices": ";".join(self.selected_vis),
                            "CV_mode": self.args.cv_mode,
                            "Group_column": self.args.group_col,
                            "N_splits_requested": self.args.n_splits,
                        })

                        # Save out-of-fold predictions with available identifiers.
                        if test_idx is not None:
                            metadata = self.df.iloc[test_idx]
                            pred_data = {
                                "Experiment": exp_name,
                                "Split": split_name,
                                "Model": model_name,
                                "Target": self.args.target,
                                "Feature_set": self.args.feature_set,
                                "Stats": self.args.stats,
                                "Include_canopy": self.args.include_canopy,
                                "Selected_stages": ";".join(self.selected_stages),
                                "Vegetation_indices": ";".join(self.selected_vis),
                                "CV_mode": self.args.cv_mode,
                                "Group_column": self.args.group_col,
                                "N_splits_requested": self.args.n_splits,
                            }
                            for col in ("location", "trial", "plot_id", "genotype"):
                                if col in metadata.columns:
                                    pred_data[col] = metadata[col].astype(str).values

                            # Preserve a custom grouping column if it is not one of
                            # the standard identifiers above.
                            if self.args.group_col not in pred_data:
                                pred_data[self.args.group_col] = (
                                    metadata[self.args.group_col].astype(str).values
                                )

                            pred_data["Actual"] = y_test
                            pred_data["Predicted"] = y_pred
                            pred_df = pd.DataFrame(pred_data)
                            self.predictions.append(pred_df)

                        plot_actual_vs_pred(
                            y_test, y_pred,
                            os.path.join(split_dir, f"{model_name}_actual_vs_predicted.png"),
                            f"{model_name}: actual vs. predicted\n(R²: {r2:.3f})",
                        )

                        if importance is not None:
                            if len(importance) != len(exp_feature_cols):
                                raise ValueError(
                                    f"Importance length mismatch: {len(importance)} vs {len(exp_feature_cols)} features"
                                )
                            imp_df = pd.DataFrame({
                                "Feature": exp_feature_cols,
                                "Importance": importance,
                            })
                            plot_feature_importance(
                                imp_df,
                                os.path.join(split_dir, f"{model_name}_feature_importance.png"),
                                f"{model_name}: top 15 feature importances",
                            )

                        with open(self.report_path, "a", encoding="utf-8") as f:
                            f.write(f"#### Model: {model_name}\n")
                            f.write(f"- **R²:** {r2:.4f}\n")
                            f.write(f"- **RMSE:** {rmse:.4f}\n")
                            f.write(f"- **MAE:** {mae:.4f}\n")
                            f.write(
                                f"![Actual vs. Predicted](./{exp_name}/{split_name}/{model_name}_actual_vs_predicted.png)\n"
                            )
                            if importance is not None:
                                f.write(
                                    f"![Feature Importance](./{exp_name}/{split_name}/{model_name}_feature_importance.png)\n"
                                )
                            f.write("\n")

                        print(f"✓ (R²={r2:.3f}, RMSE={rmse:.3f})")

                    except Exception as e:
                        print(f"✗ FAILED: {e}")
                        with open(self.report_path, "a", encoding="utf-8") as f:
                            f.write(f"#### Model: {model_name}\n")
                            f.write("- **Status:** FAILED\n")
                            f.write(f"- **Reason:** `{e}`\n\n")

        self._finalize()

    def _finalize(self):
        print(f"\n{'=' * 70}")
        print("Finalizing report...")
        print(f"{'=' * 70}\n")

        if not self.results:
            print("No models completed successfully.")
            return

        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(self.output_dir, "results_summary.csv")
        results_df.to_csv(results_path, index=False)

        predictions_path = None
        if len(self.predictions) > 0:
            predictions_df = pd.concat(self.predictions, ignore_index=True)
            predictions_path = os.path.join(self.output_dir, "predictions.csv")
            predictions_df.to_csv(predictions_path, index=False)

        # Aggregate across splits. For random mode, this is just one split.
        summary_df = (
            results_df
            .groupby(["Experiment", "Model"], as_index=False)
            .agg(
                R2_mean=("R2", "mean"),
                R2_std=("R2", "std"),
                RMSE_mean=("RMSE", "mean"),
                RMSE_std=("RMSE", "std"),
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
                N_splits=("Split", "nunique"),
            )
        )
        summary_path = os.path.join(self.output_dir, "results_summary_mean.csv")
        summary_df.to_csv(summary_path, index=False)

        # Comparison plot uses mean metrics across splits
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        sns.barplot(
            x="Experiment", y="R2_mean", hue="Model",
            data=summary_df, ax=axes[0], palette="magma"
        )
        axes[0].set_title("Model comparison by mean R²", fontsize=16)
        axes[0].tick_params(axis="x", rotation=45)
        axes[0].set_ylabel("Mean R² score")

        sns.barplot(
            x="Experiment", y="RMSE_mean", hue="Model",
            data=summary_df, ax=axes[1], palette="magma"
        )
        axes[1].set_title("Model comparison by mean RMSE", fontsize=16)
        axes[1].tick_params(axis="x", rotation=45)
        axes[1].set_ylabel("Mean root mean squared error")

        fig.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "performance_comparison.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

        best_r2 = summary_df.loc[summary_df["R2_mean"].idxmax()]
        best_rmse = summary_df.loc[summary_df["RMSE_mean"].idxmin()]

        with open(self.report_path, "a", encoding="utf-8") as f:
            f.write("# Final summary & conclusion\n\n")
            f.write("![Performance Comparison](./performance_comparison.png)\n\n")
            f.write("## Overall best performing models\n\n")
            f.write("**Highest mean R² score:**\n")
            f.write(f"- **Model:** {best_r2['Model']}\n")
            f.write(f"- **Experiment:** {best_r2['Experiment']}\n")
            f.write(f"- **Mean R²:** {best_r2['R2_mean']:.4f}\n")
            if not pd.isna(best_r2["R2_std"]):
                f.write(f"- **SD R²:** {best_r2['R2_std']:.4f}\n")
            f.write("\n")

            f.write("**Lowest mean RMSE:**\n")
            f.write(f"- **Model:** {best_rmse['Model']}\n")
            f.write(f"- **Experiment:** {best_rmse['Experiment']}\n")
            f.write(f"- **Mean RMSE:** {best_rmse['RMSE_mean']:.4f}\n")
            if not pd.isna(best_rmse["RMSE_std"]):
                f.write(f"- **SD RMSE:** {best_rmse['RMSE_std']:.4f}\n")
            f.write("\n")

            f.write("## Output tables\n\n")
            f.write(f"- Per-split results: `{os.path.basename(results_path)}`\n")
            f.write(f"- Mean results: `{os.path.basename(summary_path)}`\n")
            if predictions_path is not None:
                f.write(f"- Predictions: `{os.path.basename(predictions_path)}`\n")
            f.write("\n")

            f.write("## Interpretation\n\n")
            f.write(
                f"Based on the analysis, the **{best_r2['Model']}** model applied to "
                f"the **{best_r2['Experiment']}** experiment provided the best predictive "
                f"performance for '{self.args.target}' in terms of mean R².\n"
            )

        print(f"✓ Report saved to: {self.report_path}")
        print(f"✓ Per-split results saved to: {results_path}")
        print(f"✓ Mean results saved to: {summary_path}")
        if predictions_path is not None:
            print(f"✓ Predictions saved to: {predictions_path}")
        print(f"✓ Plots saved to: {self.output_dir}/")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate ML models for trait prediction (old-style outputs + CV modes)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Input CSV (imputed features)")
    parser.add_argument("--target", required=True, help="Numeric target column")
    parser.add_argument("--output-dir", default="ml_results")

    parser.add_argument("--models", nargs="+",
                        default=["Lasso", "Ridge", "RF", "XGBoost", "CNN"],
                        choices=["Lasso", "Ridge", "RF", "XGBoost", "CNN"])

    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--cnn-epochs", type=int, default=100)

    # CV args
    parser.add_argument("--cv-mode",
                        choices=["random", "hybrid_groupkfold"],
                        default="random",
                        help="Validation mode: random split or GroupKFold by a grouping column.")
    parser.add_argument(
        "--group-col", "--hybrid-col",
        dest="group_col",
        default="genotype",
        help=("Column used to keep complete groups together in GroupKFold. "
              "--hybrid-col is retained as a backward-compatible alias."),
    )
    parser.add_argument("--n-splits",
                        type=int,
                        default=5,
                        help="Number of folds for GroupKFold.")

    # feature selection args
    parser.add_argument("--feature-set",
                        choices=["all", "veg_qgis", "veg_exg80", "combined"],
                        default="veg_qgis",
                        help="Which features to use based on mask suffixes.")
    parser.add_argument("--stats",
                        choices=["mean", "std", "mean_std"],
                        default="mean_std",
                        help="Which stats to use for VI columns.")
    parser.add_argument("--include-canopy",
                        action="store_true",
                        help="Include canopy cover columns.")
    parser.add_argument("--include-regex",
                        default=None,
                        help="Regex filter to KEEP only matching features.")
    parser.add_argument("--exclude-regex",
                        default=None,
                        help="Regex filter to DROP matching features.")
    parser.add_argument(
        "--vegetation-indices",
        nargs="+",
        default=None,
        help=("Optional exact VI names to keep, e.g. NDVI NDRE GNDVI. "
              "Matching is case-insensitive. Without this option, all VIs are used."),
    )
    parser.add_argument("--show-feature-summary",
                        action="store_true",
                        help="Print a summary of selected features.")

    parser.add_argument("--stages", nargs="+", default=None,
                        help=("Optional list of exact growth-stage names to keep. "
                              "Example: --stages Emerg Veg_mid_1 Flow_full_2 Grain_blister_2 Mat_physiol_2. "
                              "Matching is case-insensitive but exact by full stage name."))

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    print("=" * 70)
    print("PhenoMaize GroupKFold Prediction Pipeline")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Target: {args.target}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Output: {args.output_dir}/")
    print(f"CV mode: {args.cv_mode}")
    print(f"Feature-set: {args.feature_set} | stats: {args.stats} | canopy: {args.include_canopy}")
    if args.cv_mode == "hybrid_groupkfold":
        print(f"Group column: {args.group_col} | n_splits: {args.n_splits}")
    if args.include_regex:
        print(f"Include regex: {args.include_regex}")
    if args.exclude_regex:
        print(f"Exclude regex: {args.exclude_regex}")
    if args.vegetation_indices:
        print(f"Vegetation indices: {', '.join(args.vegetation_indices)}")
    if args.stages:
        print(f"Growth stages: {', '.join(args.stages)}")
    else:
        print("Growth stages: all available stages")
    print()

    pipeline = MLPipeline(args)
    pipeline.run()

    print("\n" + "=" * 70)
    print("Pipeline complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
