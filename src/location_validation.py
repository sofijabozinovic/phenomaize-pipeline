#!/usr/bin/env python3
"""
PhenoMaize ML module (flexible feature selection + OLD output structure)

OUTPUT (like the old script):
- <output_dir>/<Experiment>/  (plots + feature importance where applicable)
- <output_dir>/performance_comparison.png
- <output_dir>/report.md

Supports:
- feature-set: all / veg_qgis / veg_exg80 / combined
- stats: mean / std / mean_std
- optional canopy cover features
- include/exclude regex filters
- target leakage protection
- models: Lasso, Ridge, RF, XGBoost, CNN (CNN trains+evaluates but NO feature importance)
- experiments: automatically inferred from feature column prefixes (generalized)
"""

import os
import sys
import argparse
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
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
    'location',
    'material_type',
    'plot_id',
    'fao',
    'rep',
    'genotype',
    'trial'
}

TARGETS = ['yield', 'moisture', 'dta']
TARGET_COLUMNS = set(TARGETS)


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
            nn.Linear(32, 1)
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
    """All numeric columns except blacklist + ALL targets (prevents leakage)."""
    num_cols = df.select_dtypes(include="number").columns.tolist()
    feats = [
        c for c in num_cols
        if (c not in BLACKLIST)
        and (c not in TARGET_COLUMNS)
        and (c != target)
    ]
    leaked = set(feats) & TARGET_COLUMNS
    if leaked:
        raise RuntimeError(f"Leakage: target columns present in features: {sorted(leaked)}")
    return feats


def select_features(df, target, feature_set="veg_qgis", stats="mean_std",
                    include_canopy=False, include_regex=None, exclude_regex=None,
                    show_summary=False):
    """
    feature_set:
      - all: columns ending _mean_all/_std_all
      - veg_qgis: columns ending _mean_veg_qgis/_std_veg_qgis
      - veg_exg80: columns ending _mean_veg_exg80/_std_veg_exg80
      - combined: all + veg_qgis (+ optional veg_exg80 only if you add it below)

    stats:
      - mean
      - std
      - mean_std
    """
    feats = _base_numeric_features(df, target)

    want_mean = stats in ("mean", "mean_std")
    want_std = stats in ("std", "mean_std")

    keep = []

    def add_suffix(suf_mean, suf_std):
        if want_mean:
            keep.extend([c for c in feats if c.endswith(suf_mean)])
        if want_std:
            keep.extend([c for c in feats if c.endswith(suf_std)])

    if feature_set == "all":
        add_suffix("_mean_all", "_std_all")
    elif feature_set == "veg_qgis":
        add_suffix("_mean_veg_qgis", "_std_veg_qgis")
    elif feature_set == "veg_exg80":
        add_suffix("_mean_veg_exg80", "_std_veg_exg80")
    elif feature_set == "combined":
        add_suffix("_mean_all", "_std_all")
        add_suffix("_mean_veg_qgis", "_std_veg_qgis")
        # ako želiš i exg80 u combined, odkomentariši:
        # add_suffix("_mean_veg_exg80", "_std_veg_exg80")
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    if include_canopy:
        # canopy se često ne završava sa mean/std, pa ga hvataš substring-om
        if feature_set in ("veg_qgis", "combined"):
            keep.extend([c for c in feats if "canopy_cover_qgis" in c])
        if feature_set in ("veg_exg80", "all", "combined"):
            keep.extend([c for c in feats if "canopy_cover_exg80" in c])

    # de-duplicate preserving order
    keep = list(dict.fromkeys(keep))

    # include/exclude regex filters (applied last)
    if include_regex:
        rgx = re.compile(include_regex)
        keep = [c for c in keep if rgx.search(c)]
    if exclude_regex:
        rgx = re.compile(exclude_regex)
        keep = [c for c in keep if not rgx.search(c)]

    if show_summary:
        print("\nFeature selection summary:")
        print(f"  feature_set={feature_set}, stats={stats}, include_canopy={include_canopy}")
        print(f"  include_regex={include_regex}, exclude_regex={exclude_regex}")
        print(f"  total selected: {len(keep)}")
        def count_contains(s): return sum(1 for c in keep if s in c)
        print(f"  contains veg_qgis: {count_contains('_veg_qgis')}")
        print(f"  contains veg_exg80: {count_contains('_veg_exg80')}")
        print(f"  contains _all: {count_contains('_all')}")
        print(f"  canopy_qgis cols: {count_contains('canopy_cover_qgis')}")
        print(f"  canopy_exg80 cols: {count_contains('canopy_cover_exg80')}\n")

    if len(keep) == 0:
        raise RuntimeError("No features selected. Check feature_set/stats/regex filters.")

    return keep


# -----------------------------
# Experiments (generalized)
# -----------------------------
def define_experiments(feature_cols):
    """
    Generalized experiment definition:
    experiment = prefix before first "_"
    plus All_Phases = all selected features
    """
    experiments = {}
    for col in feature_cols:
        if "_" in col:
            exp = col.split("_", 1)[0]
        else:
            exp = "Unknown"
        experiments.setdefault(exp, []).append(col)

    experiments["All_Phases"] = feature_cols
    return experiments


# -----------------------------
# Reciprocal location validation
# -----------------------------
class MLPipeline:
    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        self.report_path = os.path.join(self.output_dir, "location_validation_report.md")
        os.makedirs(self.output_dir, exist_ok=True)

        print(f"Loading data from {args.input}...")
        self.df = pd.read_csv(args.input)

        required = {"location", args.target}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.df["location"] = self.df["location"].astype(str).str.strip()
        self.df = self.df[self.df["location"].ne("")].copy()

        before = len(self.df)
        self.df = self.df.dropna(subset=[args.target]).copy()
        if len(self.df) < before:
            print(f"Warning: dropped {before - len(self.df)} rows with missing target values.")

        self.locations = sorted(self.df["location"].unique().tolist())
        if len(self.locations) != 2:
            raise ValueError(
                "Reciprocal location validation requires exactly two locations. "
                f"Found {len(self.locations)}: {self.locations}"
            )

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

        self.feature_cols = self._filter_stages(self.feature_cols, args.stages)
        self.feature_cols = self._filter_vegetation_indices(
            self.feature_cols, args.vegetation_indices
        )

        if not self.feature_cols:
            raise RuntimeError("No features remain after filtering.")

        self.experiments = define_experiments(self.feature_cols)
        self.results = []
        self.predictions = []
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Locations: {self.locations}")
        print(f"Target: {args.target}")
        print(f"Selected features: {len(self.feature_cols)}")
        print(f"Experiments: {list(self.experiments)}")
        print(f"Device: {self.device.upper()}")
        print(f"Output directory: {self.output_dir}")

        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(f"# Reciprocal location validation: `{self.args.target}`\n\n")
            f.write(f"Input file: `{os.path.basename(self.args.input)}`\n\n")
            f.write(f"Locations: `{', '.join(self.locations)}`\n\n")
            f.write("## Feature selection\n\n")
            f.write(f"- feature_set: `{self.args.feature_set}`\n")
            f.write(f"- stats: `{self.args.stats}`\n")
            f.write(f"- include_canopy: `{self.args.include_canopy}`\n")
            f.write(f"- stages: `{self.args.stages}`\n")
            f.write(f"- vegetation_indices: `{self.args.vegetation_indices}`\n")
            f.write(f"- selected features: `{len(self.feature_cols)}`\n\n")

    @staticmethod
    def _filter_stages(feature_cols, stages):
        if not stages:
            return feature_cols
        wanted = [s.strip().lower() for s in stages if s.strip()]
        selected = [c for c in feature_cols if any(s in c.lower() for s in wanted)]
        print(f"Stage filter retained {len(selected)} of {len(feature_cols)} features.")
        return selected

    @staticmethod
    def _filter_vegetation_indices(feature_cols, vegetation_indices):
        if not vegetation_indices:
            return feature_cols
        wanted = {v.strip().lower() for v in vegetation_indices if v.strip()}
        selected = []
        for col in feature_cols:
            first_token = col.split("_", 1)[0].lower()
            if first_token in wanted:
                selected.append(col)
        print(
            "Vegetation-index filter retained "
            f"{len(selected)} of {len(feature_cols)} features."
        )
        return selected

    def _train_cnn(self, X_train, y_train, X_test):
        model = PyTorchCNN(X_train.shape[1]).to(self.device)
        Xtr = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        ytr = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        Xte = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=32, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.MSELoss()

        model.train()
        for _ in range(self.args.cnn_epochs):
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            return model(Xte).cpu().numpy().reshape(-1)

    def _fit_predict(self, model_name, X_train, y_train, X_test):
        if model_name == "CNN":
            return self._train_cnn(X_train, y_train, X_test), None

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

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test).reshape(-1)

        importance = None
        if model_name in ("RF", "XGBoost"):
            importance = model.feature_importances_
        elif model_name in ("Lasso", "Ridge"):
            importance = np.abs(model.coef_)

        return y_pred, importance

    def run_location_validation(self, train_location, test_location):
        validation_name = f"train_{train_location}_test_{test_location}"
        train_df = self.df[self.df["location"] == train_location].copy()
        test_df = self.df[self.df["location"] == test_location].copy()

        print(f"\n{'#' * 70}")
        print(f"Validation: {train_location} -> {test_location}")
        print(f"Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        print(f"{'#' * 70}")

        with open(self.report_path, "a", encoding="utf-8") as f:
            f.write(f"## {train_location} → {test_location}\n\n")
            f.write(f"- Train rows: `{len(train_df)}`\n")
            f.write(f"- Test rows: `{len(test_df)}`\n\n")

        for exp_name, exp_features in self.experiments.items():
            exp_dir = os.path.join(self.output_dir, validation_name, exp_name)
            os.makedirs(exp_dir, exist_ok=True)

            X_train = train_df[exp_features].to_numpy()
            y_train = train_df[self.args.target].to_numpy().reshape(-1)
            X_test = test_df[exp_features].to_numpy()
            y_test = test_df[self.args.target].to_numpy().reshape(-1)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            with open(self.report_path, "a", encoding="utf-8") as f:
                f.write(f"### Experiment: {exp_name}\n\n")
                f.write(f"Features: `{len(exp_features)}`\n\n")

            for model_name in self.args.models:
                print(f"  {exp_name} | {model_name}...", end=" ")
                try:
                    y_pred, importance = self._fit_predict(
                        model_name,
                        X_train_scaled,
                        y_train,
                        X_test_scaled,
                    )

                    r2 = r2_score(y_test, y_pred)
                    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
                    mae = float(mean_absolute_error(y_test, y_pred))

                    self.results.append({
                        "Validation": validation_name,
                        "Train_location": train_location,
                        "Test_location": test_location,
                        "Experiment": exp_name,
                        "Model": model_name,
                        "N_train": len(train_df),
                        "N_test": len(test_df),
                        "N_features": len(exp_features),
                        "R2": r2,
                        "RMSE": rmse,
                        "MAE": mae,
                    })

                    metadata_cols = [
                        c for c in [
                            "location", "material_type", "plot_id", "fao",
                            "rep", "genotype", "trial"
                        ] if c in test_df.columns
                    ]
                    for row_idx, (actual, predicted) in enumerate(zip(y_test, y_pred)):
                        row = {
                            "Validation": validation_name,
                            "Train_location": train_location,
                            "Test_location": test_location,
                            "Experiment": exp_name,
                            "Model": model_name,
                            "Actual": actual,
                            "Predicted": predicted,
                            "Residual": actual - predicted,
                        }
                        for col in metadata_cols:
                            row[col] = test_df.iloc[row_idx][col]
                        self.predictions.append(row)

                    pred_plot = os.path.join(
                        exp_dir, f"{model_name}_actual_vs_predicted.png"
                    )
                    plot_actual_vs_pred(
                        y_test,
                        y_pred,
                        pred_plot,
                        f"{model_name}: {train_location} → {test_location}\nR²={r2:.3f}",
                    )

                    if importance is not None:
                        imp_df = pd.DataFrame({
                            "Feature": exp_features,
                            "Importance": importance,
                        })
                        imp_df.to_csv(
                            os.path.join(exp_dir, f"{model_name}_feature_importance.csv"),
                            index=False,
                        )
                        plot_feature_importance(
                            imp_df,
                            os.path.join(
                                exp_dir, f"{model_name}_feature_importance.png"
                            ),
                            f"{model_name}: top feature importances\n{train_location} → {test_location}",
                        )

                    with open(self.report_path, "a", encoding="utf-8") as f:
                        f.write(f"#### {model_name}\n\n")
                        f.write(f"- R²: `{r2:.4f}`\n")
                        f.write(f"- RMSE: `{rmse:.4f}`\n")
                        f.write(f"- MAE: `{mae:.4f}`\n\n")
                        f.write(
                            f"![Actual vs predicted](./{validation_name}/{exp_name}/"
                            f"{model_name}_actual_vs_predicted.png)\n\n"
                        )

                    print(f"OK (R²={r2:.3f}, RMSE={rmse:.3f})")
                except Exception as exc:
                    print(f"FAILED: {exc}")
                    with open(self.report_path, "a", encoding="utf-8") as f:
                        f.write(f"#### {model_name}\n\n")
                        f.write(f"FAILED: `{exc}`\n\n")

    def run(self):
        first, second = self.locations
        self.run_location_validation(first, second)
        self.run_location_validation(second, first)
        self._finalize()

    def _finalize(self):
        if not self.results:
            print("No models completed successfully.")
            return

        results_df = pd.DataFrame(self.results)
        predictions_df = pd.DataFrame(self.predictions)

        results_path = os.path.join(self.output_dir, "results_summary.csv")
        predictions_path = os.path.join(self.output_dir, "predictions.csv")
        mean_path = os.path.join(self.output_dir, "results_summary_mean.csv")

        results_df.to_csv(results_path, index=False)
        predictions_df.to_csv(predictions_path, index=False)

        mean_df = (
            results_df.groupby(["Experiment", "Model"], as_index=False)
            .agg(
                R2_mean=("R2", "mean"),
                R2_std=("R2", "std"),
                RMSE_mean=("RMSE", "mean"),
                RMSE_std=("RMSE", "std"),
                MAE_mean=("MAE", "mean"),
                MAE_std=("MAE", "std"),
            )
        )
        mean_df.to_csv(mean_path, index=False)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=results_df, x="Experiment", y="R2", hue="Model")
        plt.title("Reciprocal location validation: R²")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.output_dir, "performance_comparison_r2.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

        best_r2 = results_df.loc[results_df["R2"].idxmax()]
        best_rmse = results_df.loc[results_df["RMSE"].idxmin()]

        with open(self.report_path, "a", encoding="utf-8") as f:
            f.write("# Final summary\n\n")
            f.write("![R² comparison](./performance_comparison_r2.png)\n\n")
            f.write("## Best result by R²\n\n")
            f.write(f"- Validation: `{best_r2['Validation']}`\n")
            f.write(f"- Experiment: `{best_r2['Experiment']}`\n")
            f.write(f"- Model: `{best_r2['Model']}`\n")
            f.write(f"- R²: `{best_r2['R2']:.4f}`\n\n")
            f.write("## Best result by RMSE\n\n")
            f.write(f"- Validation: `{best_rmse['Validation']}`\n")
            f.write(f"- Experiment: `{best_rmse['Experiment']}`\n")
            f.write(f"- Model: `{best_rmse['Model']}`\n")
            f.write(f"- RMSE: `{best_rmse['RMSE']:.4f}`\n")

        print(f"\nResults saved to: {self.output_dir}")
        print(f"- {results_path}")
        print(f"- {mean_path}")
        print(f"- {predictions_path}")
        print(f"- {self.report_path}")


# -----------------------------
# CLI
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Reciprocal location validation for PhenoMaize ML models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", required=True, help="Input CSV with imputed features")
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--output-dir", default="results/location_validation")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Lasso", "Ridge", "RF", "XGBoost", "CNN"],
        choices=["Lasso", "Ridge", "RF", "XGBoost", "CNN"],
    )
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--cnn-epochs", type=int, default=100)

    parser.add_argument(
        "--feature-set",
        choices=["all", "veg_qgis", "veg_exg80", "combined"],
        default="veg_qgis",
    )
    parser.add_argument(
        "--stats", choices=["mean", "std", "mean_std"], default="mean_std"
    )
    parser.add_argument("--include-canopy", action="store_true")
    parser.add_argument("--include-regex", default=None)
    parser.add_argument("--exclude-regex", default=None)
    parser.add_argument("--show-feature-summary", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Optional case-insensitive growth-stage substrings",
    )
    parser.add_argument(
        "--vegetation-indices",
        nargs="+",
        default=None,
        help="Optional exact VI names, e.g. NDVI NDRE GNDVI",
    )

    args = parser.parse_args()
    if not os.path.exists(args.input):
        parser.error(f"Input file not found: {args.input}")

    pipeline = MLPipeline(args)
    pipeline.run()


if __name__ == "__main__":
    main()
