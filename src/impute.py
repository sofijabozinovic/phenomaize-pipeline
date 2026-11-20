#!/usr/bin/env python3
"""PHENO_MaizE: conservative MICE/KNN imputation of observed UAV features.

Default: fit independently within each location; completely unobserved
location-feature combinations remain NaN. No interpolation or extrapolation.
This is a descriptive preprocessing utility: for predictive cross-validation,
fit the imputer on each training fold only (and transform validation folds).
"""
import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning

EXCLUDED_TOKENS = ('fao', 'rep', 'moisture', 'yield', 'dta', 'pltwt', 'weight',
                   'trial', 'location', 'genotype', 'hybrid', 'plot_id',
                   'material_type', 'orthomosaic', 'growth_stage', 'date', 'year')


def eligible(name):
    n = str(name).lower()
    return (not any(token in n for token in EXCLUDED_TOKENS)
            and ('_mean_' in n or n.endswith('_mean') or '_std_' in n
                 or n.endswith('_std') or 'canopy_cover' in n))


class Imputer:
    def __init__(self, filepath, method='MICE', group_by_cols=None, sep=None,
                 location_col='location', global_fit=False, **legacy_kwargs):
        if legacy_kwargs.get('interpolation', 'none') != 'none':
            raise ValueError('Interpolation is disabled in this validated-methods version.')
        self.df = pd.read_csv(filepath, sep=sep or None, engine='c' if sep else 'python')
        if self.df.shape[1] < 2:
            raise ValueError('Only one CSV column detected; check --sep.')
        self.method = method.upper()
        if self.method not in ('MICE', 'KNN'):
            raise ValueError('Method must be MICE or KNN.')
        self.location_col = location_col
        if not global_fit and location_col not in self.df:
            raise ValueError(f'Missing location column {location_col!r}; use --global-fit only if appropriate.')
        self.global_fit = global_fit
        # --group-by preserves the old sequential-pass interface; each pass
        # remains nested within location unless --global-fit is explicit.
        self.group_by_cols = list(group_by_cols or [])
        for col in self.group_by_cols:
            if col not in self.df:
                raise ValueError(f'Grouping column not found: {col}')
        self.features = [c for c in self.df.columns if eligible(c)]
        for col in self.features:
            self.df[col] = pd.to_numeric(
                self.df[col].astype('string').str.replace(',', '.', regex=False),
                errors='coerce')
        self.features = [c for c in self.features if pd.api.types.is_numeric_dtype(self.df[c])]
        self.targets = [c for c in self.features if self.df[c].isna().any()]
        self.observed = self.df[self.targets].copy()
        self.missing_mask = self.observed.isna()
        self.flags = pd.DataFrame(0, index=self.df.index, columns=self.targets, dtype='int8')
        self.events = []
        print(f'Loaded {len(self.df)} rows; {len(self.features)} UAV features; '
              f'{int(self.missing_mask.sum().sum())} missing values.')

    def _fit_group(self, idx, max_iter, n_neighbors, random_state):
        missing = self.df.loc[idx, self.targets].isna()
        if len(idx) < 2 or not missing.to_numpy().any():
            return 0, 0, False
        # All-NaN columns in a group are not estimable from that group.
        usable = [c for c in self.features if self.df.loc[idx, c].notna().any()]
        if not usable:
            return 0, 0, False
        matrix = self.df.loc[idx, usable]
        scaler = StandardScaler()
        scaled = scaler.fit_transform(matrix)
        model = (IterativeImputer(max_iter=max_iter, random_state=random_state)
                 if self.method == 'MICE' else KNNImputer(n_neighbors=n_neighbors))
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always', ConvergenceWarning)
            transformed = model.fit_transform(scaled)
        elapsed = time.perf_counter() - start
        if transformed.shape[1] != len(usable):
            raise RuntimeError('Unexpected number of output columns.')
        result = pd.DataFrame(scaler.inverse_transform(transformed), index=idx, columns=usable)
        filled = 0
        for col in self.targets:
            if col not in result:
                continue
            to_fill = missing.index[missing[col]]
            values = result.loc[to_fill, col].dropna()
            if not values.empty:
                self.df.loc[values.index, col] = values
                self.flags.loc[values.index, col] = 1
                filled += len(values)
        converged_warning = any(issubclass(w.category, ConvergenceWarning) for w in captured)
        n_iter = getattr(model, 'n_iter_', None)
        return filled, elapsed, converged_warning, n_iter

    def impute(self, max_iter=10, n_neighbors=5, random_state=0):
        if max_iter < 1 or n_neighbors < 1:
            raise ValueError('max_iter and n_neighbors must be >= 1.')
        if not self.targets:
            return
        passes = self.group_by_cols or [None]
        for pass_number, extra in enumerate(passes, 1):
            cols = [] if self.global_fit else [self.location_col]
            if extra and extra not in cols:
                cols.append(extra)
            grouped = ([(('all',), self.df.index)] if not cols else
                       [(key, frame.index) for key, frame in
                        self.df.groupby(cols, sort=False, dropna=False)])
            for key, idx in grouped:
                if not self.df.loc[idx, self.targets].isna().to_numpy().any():
                    continue
                filled, seconds, warn, n_iter = self._fit_group(
                    idx, max_iter, n_neighbors, random_state)
                self.events.append({'pass': pass_number, 'group': str(key),
                                    'rows': len(idx), 'filled': filled,
                                    'seconds': round(seconds, 3),
                                    'mice_iterations': n_iter,
                                    'convergence_warning': warn})
                print(f'Pass {pass_number}, {key}: filled={filled}, '
                      f'time={seconds:.1f}s, iterations={n_iter}, '
                      f'convergence_warning={warn}', flush=True)
            remaining = int(self.df[self.targets].isna().sum().sum())
            print(f'After pass {pass_number}: {remaining} missing values remain.')
            if remaining == 0:
                break
        # Flag unresolved values explicitly; protect all observed measurements.
        for col in self.targets:
            self.flags.loc[self.df[col].isna(), col] = 3
            obs = ~self.missing_mask[col]
            self.df.loc[obs, col] = self.observed.loc[obs, col]

    def summary(self):
        print('\nIMPUTATION SUMMARY')
        print('Originally missing:', int(self.missing_mask.sum().sum()))
        print('Filled by', self.method + ':', int((self.flags == 1).sum().sum()))
        print('Unfilled:', int((self.flags == 3).sum().sum()))
        print('Grouping:', 'GLOBAL (explicit)' if self.global_fit else f'within {self.location_col}')

    def save_result(self, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_csv(output_path, index=False)
        print('Saved:', output_path)

    def save_imputation_flag(self, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ids = [c for c in ('location', 'trial', 'plot_id', 'genotype') if c in self.df]
        pd.concat([self.df[ids], self.flags], axis=1).to_csv(output_path, index=False)
        print('Saved flags:', output_path, '(0=observed, 1=imputed, 3=unfilled)')

    def save_report(self, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.events, columns=['pass', 'group', 'rows', 'filled', 'seconds',
                                            'mice_iterations', 'convergence_warning']).to_csv(
            output_path, index=False)
        print('Saved report:', output_path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True)
    p.add_argument('--method', choices=['MICE', 'KNN'], default='MICE')
    p.add_argument('--output')
    p.add_argument('--flag-output')
    p.add_argument('--report-output')
    p.add_argument('--sep')
    p.add_argument('--group-by', nargs='+', help='Optional sequential grouping passes, nested in location')
    p.add_argument('--location-col', default='location')
    p.add_argument('--global-fit', action='store_true',
                   help='Explicitly allow pooled-location imputation (not recommended by default)')
    p.add_argument('--max-iter', type=int, default=10)
    p.add_argument('--neighbors', type=int, default=5)
    p.add_argument('--random-state', type=int, default=0)
    # Retained for command-line compatibility; only none is allowed.
    p.add_argument('--interpolation', choices=['none'], default='none')
    args = p.parse_args()
    base = Path(args.input)
    label = 'global' if args.global_fit else 'by_location'
    output = args.output or str(base.with_name(f'{base.stem}_imputed_{args.method.lower()}_{label}.csv'))
    stem = str(Path(output).with_suffix(''))
    imp = Imputer(args.input, method=args.method, group_by_cols=args.group_by,
                  sep='\t' if args.sep == r'\t' else args.sep,
                  location_col=args.location_col, global_fit=args.global_fit)
    imp.impute(max_iter=args.max_iter, n_neighbors=args.neighbors,
               random_state=args.random_state)
    imp.summary()
    imp.save_result(output)
    imp.save_imputation_flag(args.flag_output or stem + '_flags.csv')
    imp.save_report(args.report_output or stem + '_report.csv')


if __name__ == '__main__':
    main()
