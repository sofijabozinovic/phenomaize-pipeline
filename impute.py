#!/usr/bin/env python3
"""
PhenoMaize Imputation Module
Handles missing data using MICE or KNN with group-wise strategies.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import NotFittedError
import warnings


# Configuration
# Treat these as "do not impute" targets (by substring match, case-insensitive)
BLACKLIST = [
    'fao', 'rep',
    'moisture', 'yield', 'dta'
]

# non-numeric identifiers we never want to include as features (mostly redundant since we select numeric)
NON_NUMERIC_BLACKLIST = ['trial', 'location', 'material_type', 'genotype', 'plot_id', 'orthomosaic', 'growth_stage']


def _is_blacklisted(colname: str) -> bool:
    """Return True if column name contains any blacklist token."""
    cn = str(colname).lower()
    return any(token in cn for token in BLACKLIST)


class Imputer:
    """
    Handles data imputation with sequential, group-wise passes,
    while blacklisting specific columns from being imputation targets.
    """

    def __init__(self, filepath, method="MICE", group_by_cols=None, sep=None):
        """Initialize imputer with data and configuration."""
        try:
            if sep:
                self.df = pd.read_csv(filepath, sep=sep)
            else:
                self.df = pd.read_csv(
                    filepath,
                    sep=None,
                    engine="python"
                )

            print(f"Successfully loaded data. Shape: {self.df.shape}\n")

            # Force canopy-cover and VI statistic columns to numeric
            # in case they were read as text.
            for col in self.df.columns:
                col_lower = str(col).lower()

                if (
                    "canopy_cover" in col_lower
                    or "_mean_" in col_lower
                    or "_std_" in col_lower
                ):
                    values = (
                        self.df[col]
                        .astype(str)
                        .str.replace(",", ".", regex=False)
                    )

                    self.df[col] = pd.to_numeric(
                        values,
                        errors="coerce"
                    )

        except Exception as e:
            raise RuntimeError(
                f"Failed to read the file. Error: {e}"
            ) from e

        if self.df.shape[1] < 2:
            raise ValueError(
                "Data loaded as a single column. "
                "Check the file separator."
            )

        self.method = method.upper()
        self.group_by_cols = (
            group_by_cols
            if group_by_cols is not None
            else []
        )

        # Identify numeric feature columns
        self.feature_cols = (
            self.df
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        # Numeric columns eligible for imputation:
        # - not blacklisted
        # - contain at least one missing value
        self.cols_to_impute = [
            col
            for col in self.feature_cols
            if (
                not _is_blacklisted(col)
                and self.df[col].isnull().any()
            )
        ]

        # Number of missing values before imputation
        self.original_missing = (
            self.df[self.cols_to_impute].isnull().sum()
            if self.cols_to_impute
            else pd.Series(dtype=int)
        )

        # Store the original missing-value positions
        # for the imputation flag file.
        self.missing_mask = (
            self.df[self.cols_to_impute].isnull().copy()
            if self.cols_to_impute
            else None
        )

        if not self.cols_to_impute:
            print(
                "No target columns with missing data "
                "were found to impute."
            )
        else:
            print(
                f"Numeric feature columns: "
                f"{len(self.feature_cols)} total"
            )
            print(
                f"Target columns to impute: "
                f"{len(self.cols_to_impute)}"
            )
            print("\nOriginal missing values (only >0):")
            print(
                self.original_missing[
                    self.original_missing > 0
                ].to_string()
            )

    def _impute_subset(self, subset_df, imputer_model):
        """
        Scale, impute, and inverse-scale a data subset.
        Robust to entirely-empty columns within a group.
        """
        # Skip if no missing values (in target cols) within this subset
        if subset_df[self.cols_to_impute].isnull().sum().sum() == 0:
            return subset_df

        # Find usable numeric columns in this subset: not all-missing
        usable_feature_cols = [
            col for col in self.feature_cols
            if col in subset_df.columns and (not subset_df[col].isnull().all())
        ]

        # Need at least 2 rows and some usable features
        if not usable_feature_cols or len(subset_df) < 2:
            return subset_df

        data_for_processing = subset_df[usable_feature_cols].copy()

        # Scale
        scaler = StandardScaler()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                scaled_data = scaler.fit_transform(data_for_processing)
        except ValueError:
            scaled_data = data_for_processing.values

        # Impute (fit on subset)
        imputed_scaled_data = imputer_model.fit_transform(scaled_data)

        # Inverse transform
        try:
            imputed_original_scale_data = scaler.inverse_transform(imputed_scaled_data)
        except NotFittedError:
            imputed_original_scale_data = imputed_scaled_data

        imputed_df = pd.DataFrame(
            imputed_original_scale_data,
            columns=usable_feature_cols,
            index=subset_df.index
        )

        # Update only target columns that exist in this subset
        imputed_target_cols = [col for col in imputed_df.columns if col in self.cols_to_impute]
        if imputed_target_cols:
            subset_df.update(imputed_df[imputed_target_cols])

        return subset_df

    def _run_imputation_pass(self, imputer_model, group_by_col=None):
        """Run a single imputation pass, optionally grouped."""
        if group_by_col:
            print(f"--- Pass: grouped by '{group_by_col}' ---")
            if group_by_col not in self.df.columns:
                raise ValueError(f"Grouping column '{group_by_col}' not found.")

            imputed_groups = []
            for _, group in self.df.groupby(group_by_col, sort=False, dropna=False):
                imputed_groups.append(self._impute_subset(group.copy(), imputer_model))

            self.df = pd.concat(imputed_groups).sort_index()
        else:
            print("--- Pass: global (no grouping) ---")
            self.df = self._impute_subset(self.df, imputer_model)

        remaining_missing = self.df[self.cols_to_impute].isnull().sum().sum()
        print(f"--- Pass complete. {remaining_missing} missing values remain. ---\n")
        return remaining_missing == 0

    def impute(self, **kwargs):
        """Run the full imputation workflow."""
        if not self.cols_to_impute:
            return

        # Configure imputer
        if self.method == 'MICE':
            imputer_model = IterativeImputer(
                max_iter=kwargs.get('max_iter', 10),
                random_state=kwargs.get('random_state', 0)
            )
            print(f"\nConfigured MICE imputer (max_iter={imputer_model.max_iter})")
        elif self.method == 'KNN':
            imputer_model = KNNImputer(n_neighbors=kwargs.get('n_neighbors', 5))
            print(f"\nConfigured KNN imputer (n_neighbors={imputer_model.n_neighbors})")
        else:
            raise ValueError(f"Unknown imputation method: {self.method}")

        print(f"Planned passes: {self.group_by_cols if self.group_by_cols else ['Global']}\n")

        # Run imputation passes
        if not self.group_by_cols:
            self._run_imputation_pass(imputer_model)
        else:
            for i, group_col in enumerate(self.group_by_cols):
                print(f"Starting Pass {i+1}/{len(self.group_by_cols)}")
                all_imputed = self._run_imputation_pass(imputer_model, group_by_col=group_col)
                if all_imputed:
                    print("All target values imputed. No further passes needed.")
                    break

        print("Imputation process complete.")

    def save_result(self, output_path):
        """Save imputed data to CSV."""
        try:
            self.df.to_csv(output_path, index=False, sep=',')
            print(f"\nSaved imputed data to: {output_path}")
        except Exception as e:
            print(f"ERROR: Could not save file. {e}")

    def save_imputation_flag(self, output_path):
        """Save a CSV marking which values were imputed (1) vs original (0)."""
        if not self.cols_to_impute:
            print("No imputation performed, skipping flag file.")
            return
        if self.missing_mask is None:
            print("Missing mask not available, skipping flag file.")
            return

        # Flag DF: 1 where originally missing, else 0
        flag_df = pd.DataFrame(index=self.df.index)

        # keep identifiers if present (helps join back)
        keep_id_cols = []
        for c in ['location', 'trial', 'plot_id', 'genotype']:
            if c in self.df.columns:
                keep_id_cols.append(c)
                flag_df[c] = self.df[c]

        # flags for each imputed column
        for col in self.cols_to_impute:
            flag_df[col] = self.missing_mask[col].astype(int)

        try:
            flag_df.to_csv(output_path, index=False, sep=',')
            print(f"Saved imputation flags to: {output_path}")
        except Exception as e:
            print(f"ERROR: Could not save flag file. {e}")

    def summary(self):
        """Print imputation summary."""
        print("\n" + "=" * 70)
        print("IMPUTATION SUMMARY")
        print("=" * 70)

        if not self.cols_to_impute:
            print("No imputation was performed.")
            return

        final_missing = self.df[self.cols_to_impute].isnull().sum()

        if final_missing.sum() == 0:
            print("✓ All missing values successfully imputed")
        else:
            print("⚠ Some missing values remain:")
            print(final_missing[final_missing > 0].to_string())

        print("\nBefore/After comparison:")
        comparison = pd.DataFrame({
            'Column': self.cols_to_impute,
            'Missing Before': [int(self.original_missing[col]) for col in self.cols_to_impute],
            'Missing After': [int(final_missing[col]) for col in self.cols_to_impute]
        })
        print(comparison.to_string(index=False))
        print("=" * 70)


def main():
    """Main imputation workflow."""
    parser = argparse.ArgumentParser(
        description='Impute missing vegetation index data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Single pass with MICE
  python impute.py --input results.csv --method MICE

  # Two-pass grouped imputation
  python impute.py --input results.csv --method MICE --group-by trial fao

  # KNN imputation
  python impute.py --input results.csv --method KNN --neighbors 5
        """
    )

    parser.add_argument('--input', required=True, help='Input CSV file with missing data')
    parser.add_argument('--method', choices=['MICE', 'KNN'], default='MICE',
                        help='Imputation method (default: MICE)')
    parser.add_argument('--output',
                        help='Output filename (default: <input>_imputed_<method>_<passes>.csv)')
    parser.add_argument('--flag-output',
                        help='Imputation flag output filename (default: <input>_imputed_<...>_flags.csv)')
    parser.add_argument('--sep', default=None,
                        help='File separator (auto-detected if not specified)')
    parser.add_argument('--group-by', nargs='+', default=None,
                        help='Columns for sequential imputation passes (e.g., trial fao)')
    parser.add_argument('--max-iter', type=int, default=10,
                        help='Max iterations for MICE (default: 10)')
    parser.add_argument('--neighbors', type=int, default=5,
                        help='Number of neighbors for KNN (default: 5)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)

    # Handle separator
    sep = args.sep
    if sep == '\\t':
        sep = '\t'

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base, _ = os.path.splitext(os.path.basename(args.input))
        passes = f"{len(args.group_by)}pass" if args.group_by else 'global'
        output_filename = f"{base}_imputed_{args.method.lower()}_{passes}.csv"
        output_path = os.path.join(os.path.dirname(args.input), output_filename)

    # Determine flag output path
    if args.flag_output:
        flag_path = args.flag_output
    else:
        base, _ = os.path.splitext(output_path)
        flag_path = f"{base}_flags.csv"

    print("=" * 70)
    print("PhenoMaize Data Imputation")
    print("=" * 70)
    print(f"Input: {args.input}")
    print(f"Method: {args.method}")
    print(f"Output: {output_path}")
    print(f"Flag file: {flag_path}")
    if args.group_by:
        print(f"Grouping: {' → '.join(args.group_by)}")
    print()

    try:
        imputer = Imputer(
            args.input,
            method=args.method,
            group_by_cols=args.group_by,
            sep=sep
        )

        if imputer.cols_to_impute:
            imputer.impute(
                max_iter=args.max_iter,
                n_neighbors=args.neighbors
            )
            imputer.summary()
            imputer.save_imputation_flag(flag_path)
        else:
            print("No imputation needed. Data is complete.")

        # Always save the output dataset so the pipeline has
        # a consistent input file for the next step.
        imputer.save_result(output_path)

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
