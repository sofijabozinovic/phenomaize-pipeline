#!/usr/bin/env python3
"""Selection-efficiency analysis for PhenoMaize GroupKFold predictions.

Input is ``predictions.csv`` produced by ``group_kfold_predictions.py``.
Predictions are aggregated to one row per Experiment × Model × genotype, after
which agreement between observed and predicted top/bottom selections is
calculated.
"""

import argparse
import os
import sys
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METADATA_COLUMNS = [
    "Target", "Feature_set", "Stats", "Include_canopy",
    "Selected_stages", "Vegetation_indices", "CV_mode",
    "Group_column", "N_splits_requested",
]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def validate_fractions(values: Iterable[float], option_name: str) -> list[float]:
    cleaned = []
    for value in values:
        value = float(value)
        if not 0 < value < 1:
            raise ValueError(f"{option_name} values must be between 0 and 1; got {value}.")
        if value not in cleaned:
            cleaned.append(value)
    return cleaned


def resolve_group_column(pred_df: pd.DataFrame, requested: str) -> str:
    """Resolve the genotype/group identifier with backward compatibility."""
    lookup = {str(col).lower(): col for col in pred_df.columns}
    requested_key = str(requested).lower()
    if requested_key in lookup:
        return lookup[requested_key]
    if requested_key == "genotype" and "hybrid" in lookup:
        print("Warning: using legacy 'Hybrid' column as genotype identifier.")
        return lookup["hybrid"]
    raise ValueError(
        f"Group column '{requested}' is absent from predictions.csv. "
        f"Available columns: {', '.join(map(str, pred_df.columns))}"
    )


def filter_case_insensitive(df: pd.DataFrame, column: str, values: list[str] | None) -> pd.DataFrame:
    if not values:
        return df
    requested = {str(value).strip().lower() for value in values}
    available_map = {str(value).lower(): value for value in df[column].dropna().unique()}
    missing = sorted(value for value in requested if value not in available_map)
    if missing:
        raise ValueError(
            f"Unknown values for --{column.lower()} filtering: {', '.join(missing)}. "
            f"Available: {', '.join(sorted(map(str, available_map.values())))}"
        )
    return df[df[column].astype(str).str.lower().isin(requested)].copy()


def _select_set(df: pd.DataFrame, value_col: str, fraction: float,
                mode: str, group_col: str) -> set[str]:
    """Return identifiers selected by the requested value and fraction."""
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")

    n = len(df)
    k = max(1, int(np.ceil(n * fraction)))
    ascending = mode == "bottom"
    if mode not in {"bottom", "top"}:
        raise ValueError("mode must be 'bottom' or 'top'")

    # The identifier is a deterministic secondary key when values are tied.
    selected = (
        df.sort_values([value_col, group_col], ascending=[ascending, True], kind="mergesort")
        .head(k)
    )
    return set(selected[group_col].astype(str))


def selection_metrics(df: pd.DataFrame, fraction: float, mode: str,
                      group_col: str) -> dict:
    actual_set = _select_set(df, "Actual", fraction, mode, group_col)
    pred_set = _select_set(df, "Predicted", fraction, mode, group_col)

    correct = actual_set & pred_set
    tp = len(correct)
    fp = len(pred_set - actual_set)
    fn = len(actual_set - pred_set)
    precision = tp / len(pred_set) if pred_set else np.nan
    recall = tp / len(actual_set) if actual_set else np.nan
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else np.nan)

    return {
        "Mode": mode,
        "Fraction": fraction,
        "N_total_genotypes": len(df),
        "N_selected": len(pred_set),
        "True_positive": tp,
        "False_positive": fp,
        "False_negative": fn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Actual_selected_genotypes": ";".join(sorted(actual_set)),
        "Predicted_selected_genotypes": ";".join(sorted(pred_set)),
        "Correctly_selected_genotypes": ";".join(sorted(correct)),
    }


def aggregate_predictions(pred_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    required = {"Experiment", "Split", "Model", group_col, "Actual", "Predicted"}
    missing = required - set(pred_df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions file: {sorted(missing)}")

    pred_df = pred_df.copy()
    pred_df[group_col] = pred_df[group_col].astype(str)
    group_cols = ["Experiment", "Model", group_col]

    agg_spec = {
        "Actual": ("Actual", "mean"),
        "Predicted": ("Predicted", "mean"),
        "N_records": (group_col, "size"),
    }
    for col in METADATA_COLUMNS:
        if col in pred_df.columns:
            agg_spec[col] = (col, "first")

    return pred_df.groupby(group_cols, as_index=False).agg(**agg_spec)


def plot_selection_summary(summary_df: pd.DataFrame, out_path: str,
                           metric: str = "Recall") -> None:
    for mode in summary_df["Mode"].unique():
        for frac in sorted(summary_df["Fraction"].unique()):
            sub = summary_df[(summary_df["Mode"] == mode) &
                             (summary_df["Fraction"] == frac)].copy()
            if sub.empty:
                continue
            label = f"{mode}_{int(round(frac * 100))}"
            plt.figure(figsize=(12, 7))
            experiments = list(sub["Experiment"].unique())
            models = list(sub["Model"].unique())
            x = np.arange(len(experiments))
            width = 0.8 / max(1, len(models))
            for i, model in enumerate(models):
                vals = []
                for exp in experiments:
                    row = sub[(sub["Experiment"] == exp) & (sub["Model"] == model)]
                    vals.append(float(row[metric].iloc[0]) if not row.empty else np.nan)
                plt.bar(x + i * width, vals, width, label=model)
            plt.xticks(x + width * (len(models) - 1) / 2,
                       experiments, rotation=45, ha="right")
            plt.ylabel(metric)
            plt.ylim(0, 1.05)
            plt.title(f"Selection {metric}: {mode} {int(round(frac * 100))}%")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, f"selection_{metric.lower()}_{label}.png"),
                        dpi=150, bbox_inches="tight")
            plt.close()


def plot_actual_vs_predicted_genotype(agg_df: pd.DataFrame, out_dir: str) -> None:
    for (exp, model), sub in agg_df.groupby(["Experiment", "Model"]):
        exp_dir = os.path.join(out_dir, str(exp))
        ensure_dir(exp_dir)
        y_true = sub["Actual"].to_numpy()
        y_pred = sub["Predicted"].to_numpy()
        plt.figure(figsize=(7, 7))
        plt.scatter(y_true, y_pred, alpha=0.7)
        mn = float(min(np.min(y_true), np.min(y_pred)))
        mx = float(max(np.max(y_true), np.max(y_pred)))
        plt.plot([mn, mx], [mn, mx], "--", linewidth=2)
        plt.xlabel("Actual genotype mean")
        plt.ylabel("Predicted genotype mean")
        plt.title(f"{exp} - {model}: genotype-level predictions")
        plt.tight_layout()
        safe_model = str(model).replace("/", "_")
        plt.savefig(os.path.join(exp_dir, f"{safe_model}_genotype_actual_vs_predicted.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Selection-efficiency analysis from GroupKFold predictions."
    )
    parser.add_argument("--predictions", required=True,
                        help="Path to predictions.csv from group_kfold_predictions.py")
    parser.add_argument("--output-dir", default="selection_efficiency_results")
    parser.add_argument("--group-col", default="genotype",
                        help="Genotype/group identifier column in predictions.csv.")
    parser.add_argument("--top-fractions", nargs="+", type=float,
                        default=[0.20, 0.30, 0.40])
    parser.add_argument("--bottom-fractions", nargs="+", type=float,
                        default=[0.20, 0.30, 0.40])
    parser.add_argument("--models", nargs="+", default=None,
                        help="Optional model names to include.")
    parser.add_argument("--experiments", nargs="+", default=None,
                        help="Optional experiment names, e.g. All_Selected_Stages.")
    args = parser.parse_args()

    try:
        args.top_fractions = validate_fractions(args.top_fractions, "--top-fractions")
        args.bottom_fractions = validate_fractions(args.bottom_fractions, "--bottom-fractions")
        if not os.path.exists(args.predictions):
            raise FileNotFoundError(f"Predictions file not found: {args.predictions}")

        ensure_dir(args.output_dir)
        pred_df = pd.read_csv(args.predictions)
        group_col = resolve_group_column(pred_df, args.group_col)
        pred_df = filter_case_insensitive(pred_df, "Model", args.models)
        pred_df = filter_case_insensitive(pred_df, "Experiment", args.experiments)
        if pred_df.empty:
            raise ValueError("No predictions remain after filtering.")

        agg_df = aggregate_predictions(pred_df, group_col)
        agg_path = os.path.join(args.output_dir, "genotype_level_predictions.csv")
        agg_df.to_csv(agg_path, index=False)

        rows = []
        metadata_present = [col for col in METADATA_COLUMNS if col in agg_df.columns]
        for (exp, model), sub in agg_df.groupby(["Experiment", "Model"]):
            metadata = {col: sub[col].iloc[0] for col in metadata_present}
            for mode, fractions in (("bottom", args.bottom_fractions),
                                    ("top", args.top_fractions)):
                for frac in fractions:
                    row = selection_metrics(sub, frac, mode, group_col)
                    row.update({"Experiment": exp, "Model": model})
                    row.update(metadata)
                    rows.append(row)

        summary_df = pd.DataFrame(rows)
        cols_first = [
            "Experiment", "Model", "Mode", "Fraction",
            "N_total_genotypes", "N_selected", "Precision", "Recall", "F1",
            "True_positive", "False_positive", "False_negative",
        ]
        remaining = [c for c in summary_df.columns if c not in cols_first]
        summary_df = summary_df[cols_first + remaining]
        summary_path = os.path.join(args.output_dir, "selection_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        best_recall = (summary_df
                       .sort_values(["Mode", "Fraction", "Recall", "Precision"],
                                    ascending=[True, True, False, False])
                       .groupby(["Mode", "Fraction"], as_index=False).head(1))
        best_precision = (summary_df
                          .sort_values(["Mode", "Fraction", "Precision", "Recall"],
                                       ascending=[True, True, False, False])
                          .groupby(["Mode", "Fraction"], as_index=False).head(1))
        best_recall_path = os.path.join(args.output_dir, "best_by_recall.csv")
        best_precision_path = os.path.join(args.output_dir, "best_by_precision.csv")
        best_recall.to_csv(best_recall_path, index=False)
        best_precision.to_csv(best_precision_path, index=False)

        plot_selection_summary(summary_df, args.output_dir, "Recall")
        plot_selection_summary(summary_df, args.output_dir, "Precision")
        plot_actual_vs_predicted_genotype(agg_df, args.output_dir)

        report_path = os.path.join(args.output_dir, "selection_report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# Selection efficiency report\n\n")
            f.write(f"Input predictions: `{os.path.basename(args.predictions)}`\n\n")
            f.write(f"Genotype/group column: `{group_col}`\n\n")
            f.write(f"Top fractions: `{args.top_fractions}`\n\n")
            f.write(f"Bottom fractions: `{args.bottom_fractions}`\n\n")
            if metadata_present:
                f.write("## Prediction settings\n\n")
                for col in metadata_present:
                    values = "; ".join(map(str, pred_df[col].dropna().unique()))
                    f.write(f"- {col}: `{values}`\n")
                f.write("\n")
            f.write("## Interpretation\n\n")
            f.write("- **Precision**: among genotypes selected by the model, the proportion that truly belongs to the selected group.\n")
            f.write("- **Recall**: among truly selected genotypes, the proportion recovered by the model.\n")
            f.write("- **Bottom selection**: useful for discarding the poorest genotypes.\n")
            f.write("- **Top selection**: useful for retaining the best genotypes.\n\n")
            f.write("## Output files\n\n")
            f.write(f"- Genotype-level predictions: `{os.path.basename(agg_path)}`\n")
            f.write(f"- Selection summary: `{os.path.basename(summary_path)}`\n")
            f.write(f"- Best by recall: `{os.path.basename(best_recall_path)}`\n")
            f.write(f"- Best by precision: `{os.path.basename(best_precision_path)}`\n\n")
            display_cols = ["Mode", "Fraction", "Experiment", "Model",
                            "Recall", "Precision", "F1"]
            f.write("## Best by recall\n\n")
            f.write(best_recall[display_cols].to_markdown(index=False))
            f.write("\n\n## Best by precision\n\n")
            f.write(best_precision[display_cols].to_markdown(index=False))
            f.write("\n")

        print("=" * 70)
        print("Selection efficiency analysis complete")
        print("=" * 70)
        print(f"Genotype-level predictions: {agg_path}")
        print(f"Selection summary: {summary_path}")
        print(f"Best by recall: {best_recall_path}")
        print(f"Best by precision: {best_precision_path}")
        print(f"Report: {report_path}")
        print(f"Plots saved to: {args.output_dir}")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
