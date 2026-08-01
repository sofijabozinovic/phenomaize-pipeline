#!/usr/bin/env python3
"""
PHENO_MaizE pipeline orchestrator.

Supported modules

    extract.py
    impute.py
    ml.py
    group_kfold_predictions.py
    location_validation.py
    selection_efficiency.py

The orchestrator provides a unified command-line interface
without modifying the individual analysis modules.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SCRIPT_DIR = Path(__file__).resolve().parent

EXTRACT_SCRIPT = SCRIPT_DIR / "extract.py"
IMPUTE_SCRIPT = SCRIPT_DIR / "impute.py"
ML_SCRIPT = SCRIPT_DIR / "ml.py"
GROUP_KFOLD_SCRIPT = SCRIPT_DIR / "group_kfold_predictions.py"
LOCATION_VALIDATION_SCRIPT = SCRIPT_DIR / "location_validation.py"
SELECTION_EFFICIENCY_SCRIPT = SCRIPT_DIR / "selection_efficiency.py"


def absolute_path(value: str | Path) -> Path:
    """Convert a user-supplied path into an absolute path."""
    path = Path(value).expanduser()

    if not path.is_absolute():
        path = Path.cwd() / path

    return path.resolve()


def format_command(command: Sequence[str]) -> str:
    """Format a subprocess command for readable terminal output."""
    return subprocess.list2cmdline(list(command))


def ensure_script_exists(script_path: Path) -> None:
    """Raise an error when a required pipeline script is missing."""
    if not script_path.is_file():
        raise FileNotFoundError(
            f"Required script was not found: {script_path}"
        )


def ensure_input_exists(input_path: Path, description: str) -> None:
    """Raise an error when a required input file or directory is missing."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {input_path}"
        )


def run_command(
    command: Sequence[str],
    step_name: str,
    expected_outputs: Sequence[Path],
    force: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Execute one pipeline command.

    The step is skipped when all required outputs already exist,
    unless --force is enabled.
    """
    outputs_exist = (
        bool(expected_outputs)
        and all(path.is_file() for path in expected_outputs)
    )

    if outputs_exist and not force:
        print(f"\n[SKIP] {step_name}")
        for path in expected_outputs:
            print(f"       Existing output: {path}")
        return

    print(f"\n[RUN]  {step_name}")
    print(f"       {format_command(command)}")

    if dry_run:
        print("       Dry run: command was not executed.")
        return

    subprocess.run(
        list(command),
        cwd=SCRIPT_DIR,
        check=True,
    )

    missing_outputs = [
        path for path in expected_outputs if not path.is_file()
    ]

    if missing_outputs:
        formatted = "\n".join(f"  - {path}" for path in missing_outputs)
        raise RuntimeError(
            f"{step_name} finished without producing the expected output(s):\n"
            f"{formatted}"
        )

    print(f"[DONE] {step_name}")


def build_extract_command(
    data_dir: Path,
    output_dir: Path,
    flight_info: str,
    field_data: str,
    wide_filename: str,
    long_filename: str,
) -> list[str]:
    """Construct the extract.py command."""
    return [
        sys.executable,
        str(EXTRACT_SCRIPT),
        "--data-dir",
        str(data_dir),
        "--flight-info",
        flight_info,
        "--field-data",
        field_data,
        "--results-dir",
        str(output_dir),
        "--output",
        wide_filename,
        "--intermediate",
        long_filename,
    ]


def build_impute_command(
    input_path: Path,
    method: str,
    output_path: Path,
    flag_output_path: Path,
    group_by: list[str] | None,
    max_iter: int,
    neighbors: int,
    sep: str | None,
) -> list[str]:
    """Construct the impute.py command."""
    command = [
        sys.executable,
        str(IMPUTE_SCRIPT),
        "--input",
        str(input_path),
        "--method",
        method,
        "--output",
        str(output_path),
        "--flag-output",
        str(flag_output_path),
        "--max-iter",
        str(max_iter),
        "--neighbors",
        str(neighbors),
    ]

    if sep is not None:
        command.extend(["--sep", sep])

    if group_by:
        command.append("--group-by")
        command.extend(group_by)

    return command


def append_prediction_options(
    command: list[str],
    args: argparse.Namespace,
) -> None:
    """Append optional arguments shared by prediction scripts."""

    if args.models:
        command.append("--models")
        command.extend(args.models)

    if args.alpha is not None:
        command.extend(["--alpha", str(args.alpha)])

    if args.n_estimators is not None:
        command.extend(["--n-estimators", str(args.n_estimators)])

    if args.cnn_epochs is not None:
        command.extend(["--cnn-epochs", str(args.cnn_epochs)])

    if args.feature_set is not None:
        command.extend(["--feature-set", args.feature_set])

    if args.stats is not None:
        command.extend(["--stats", args.stats])

    if args.include_canopy:
        command.append("--include-canopy")

    if args.include_regex:
        command.extend(["--include-regex", args.include_regex])

    if args.exclude_regex:
        command.extend(["--exclude-regex", args.exclude_regex])

    if args.vegetation_indices:
        command.append("--vegetation-indices")
        command.extend(args.vegetation_indices)

    if args.stages:
        command.append("--stages")
        command.extend(args.stages)

    if args.show_feature_summary:
        command.append("--show-feature-summary")

def build_ml_command(
    input_path: Path,
    target: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Construct the ml.py command."""
    command = [
        sys.executable,
        str(ML_SCRIPT),
        "--input",
        str(input_path),
        "--target",
        target,
        "--output-dir",
        str(output_dir),
    ]

    append_prediction_options(command, args)

    return command


def build_group_kfold_command(
    input_path: Path,
    target: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Construct the group_kfold_predictions.py command."""
    command = [
        sys.executable,
        str(GROUP_KFOLD_SCRIPT),
        "--input",
        str(input_path),
        "--target",
        target,
        "--output-dir",
        str(output_dir),
        "--cv-mode",
        args.cv_mode,
        "--group-col",
        args.group_col,
        "--n-splits",
        str(args.n_splits),
    ]

    append_prediction_options(command, args)

    return command


def build_location_validation_command(
    input_path: Path,
    target: str,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Construct the location_validation.py command."""
    command = [
        sys.executable,
        str(LOCATION_VALIDATION_SCRIPT),
        "--input",
        str(input_path),
        "--target",
        target,
        "--output-dir",
        str(output_dir),
    ]

    append_prediction_options(command, args)

    return command


def build_selection_efficiency_command(
    predictions_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[str]:
    """Construct the selection_efficiency.py command."""
    command = [
        sys.executable,
        str(SELECTION_EFFICIENCY_SCRIPT),
        "--predictions",
        str(predictions_path),
        "--output-dir",
        str(output_dir),
        "--group-col",
        args.group_col,
    ]

    if args.top_fractions:
        command.append("--top-fractions")
        command.extend(map(str, args.top_fractions))

    if args.bottom_fractions:
        command.append("--bottom-fractions")
        command.extend(map(str, args.bottom_fractions))

    if args.models:
        command.append("--models")
        command.extend(args.models)

    if args.experiments:
        command.append("--experiments")
        command.extend(args.experiments)

    return command


def run_extract(args: argparse.Namespace) -> None:
    """Run extract.py as an independent pipeline step."""
    ensure_script_exists(EXTRACT_SCRIPT)

    data_dir = absolute_path(args.data_dir)
    results_dir = absolute_path(args.results_dir)
    output_dir = results_dir / "extracted"

    ensure_input_exists(data_dir, "Dataset directory")

    output_dir.mkdir(parents=True, exist_ok=True)

    wide_path = output_dir / args.output
    long_path = output_dir / args.intermediate

    command = build_extract_command(
        data_dir=data_dir,
        output_dir=output_dir,
        flight_info=args.flight_info,
        field_data=args.field_data,
        wide_filename=args.output,
        long_filename=args.intermediate,
    )

    run_command(
        command=command,
        step_name="Vegetation-index extraction",
        expected_outputs=[wide_path, long_path],
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\nExtraction outputs:")
    print(f"  Wide table: {wide_path}")
    print(f"  Long table: {long_path}")


def resolve_imputation_outputs(
    input_path: Path,
    results_dir: Path,
    method: str,
    output: str | None,
    flag_output: str | None,
) -> tuple[Path, Path]:
    """Determine deterministic output paths for imputation."""
    imputed_dir = results_dir / "imputed"
    imputed_dir.mkdir(parents=True, exist_ok=True)

    method_label = method.lower()

    if output:
        output_path = absolute_path(output)
    else:
        output_path = (
            imputed_dir
            / f"{input_path.stem}_imputed_{method_label}.csv"
        )

    if flag_output:
        flag_output_path = absolute_path(flag_output)
    else:
        flag_output_path = (
            imputed_dir
            / f"{input_path.stem}_imputed_{method_label}_flags.csv"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    flag_output_path.parent.mkdir(parents=True, exist_ok=True)

    return output_path, flag_output_path


def resolve_prediction_input(
    input_value: str | None,
    results_dir: Path,
    allow_missing: bool = False,
) -> Path:
    """
    Resolve the imputed CSV used by prediction modules.

    If --input is provided, use it directly.
    Otherwise, search results/imputed for a single imputed CSV.
    """
    if input_value:
        input_path = absolute_path(input_value)

        if not allow_missing:
            ensure_input_exists(input_path, "Prediction input file")

        return input_path

    imputed_dir = results_dir / "imputed"

    if not imputed_dir.is_dir():
        raise FileNotFoundError(
            "Imputed-data directory was not found: "
            f"{imputed_dir}. Run preprocess or specify --input."
        )

    candidates = sorted(
        path
        for path in imputed_dir.glob("*.csv")
        if "imputed" in path.stem.lower()
        and not path.stem.lower().endswith("_flags")
    )

    if not candidates:
        raise FileNotFoundError(
            "No imputed CSV was found in "
            f"{imputed_dir}. Run preprocess or specify --input."
        )

    if len(candidates) > 1:
        formatted = "\n".join(f"  - {path}" for path in candidates)
        raise ValueError(
            "Multiple imputed CSV files were found. "
            "Specify one explicitly using --input:\n"
            f"{formatted}"
        )

    print(f"\nAutomatically selected prediction input: {candidates[0]}")
    return candidates[0]


def run_impute(args: argparse.Namespace) -> None:
    """Run impute.py as an independent pipeline step."""
    ensure_script_exists(IMPUTE_SCRIPT)

    input_path = absolute_path(args.input)
    results_dir = absolute_path(args.results_dir)

    ensure_input_exists(input_path, "Imputation input file")

    output_path, flag_output_path = resolve_imputation_outputs(
        input_path=input_path,
        results_dir=results_dir,
        method=args.method,
        output=args.output,
        flag_output=args.flag_output,
    )

    command = build_impute_command(
        input_path=input_path,
        method=args.method,
        output_path=output_path,
        flag_output_path=flag_output_path,
        group_by=args.group_by,
        max_iter=args.max_iter,
        neighbors=args.neighbors,
        sep=args.sep,
    )

    # Flag CSV is not required because impute.py creates it only when
    # at least one value was actually imputed.
    run_command(
        command=command,
        step_name=f"Missing-value imputation ({args.method})",
        expected_outputs=[output_path],
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\nImputation outputs:")
    print(f"  Imputed table: {output_path}")
    print(f"  Flag table:    {flag_output_path}")
    print("  Note: the flag table is produced only when imputation occurs.")


def run_preprocess(args: argparse.Namespace) -> None:
    """Run extract.py followed by impute.py."""
    ensure_script_exists(EXTRACT_SCRIPT)
    ensure_script_exists(IMPUTE_SCRIPT)

    data_dir = absolute_path(args.data_dir)
    results_dir = absolute_path(args.results_dir)

    ensure_input_exists(data_dir, "Dataset directory")

    extracted_dir = results_dir / "extracted"
    imputed_dir = results_dir / "imputed"

    extracted_dir.mkdir(parents=True, exist_ok=True)
    imputed_dir.mkdir(parents=True, exist_ok=True)

    wide_path = extracted_dir / args.output
    long_path = extracted_dir / args.intermediate

    method_label = args.method.lower()

    imputed_path = (
        imputed_dir
        / f"{wide_path.stem}_imputed_{method_label}.csv"
    )
    flag_path = (
        imputed_dir
        / f"{wide_path.stem}_imputed_{method_label}_flags.csv"
    )

    extract_command = build_extract_command(
        data_dir=data_dir,
        output_dir=extracted_dir,
        flight_info=args.flight_info,
        field_data=args.field_data,
        wide_filename=args.output,
        long_filename=args.intermediate,
    )

    run_command(
        command=extract_command,
        step_name="1/2 Vegetation-index extraction",
        expected_outputs=[wide_path, long_path],
        force=args.force,
        dry_run=args.dry_run,
    )

    # During a dry run the wide CSV does not need to exist yet because
    # the preceding extract command is also only displayed.
    if not args.dry_run:
        ensure_input_exists(wide_path, "Extracted wide table")

    impute_command = build_impute_command(
        input_path=wide_path,
        method=args.method,
        output_path=imputed_path,
        flag_output_path=flag_path,
        group_by=args.group_by,
        max_iter=args.max_iter,
        neighbors=args.neighbors,
        sep=args.sep,
    )

    run_command(
        command=impute_command,
        step_name=f"2/2 Missing-value imputation ({args.method})",
        expected_outputs=[imputed_path],
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\nPreprocessing outputs:")
    print(f"  Long table:    {long_path}")
    print(f"  Wide table:    {wide_path}")
    print(f"  Imputed table: {imputed_path}")
    print(f"  Flag table:    {flag_path}")


def run_ml(args: argparse.Namespace) -> None:
    """Run ml.py on an existing imputed dataset."""
    ensure_script_exists(ML_SCRIPT)

    input_path = absolute_path(args.input)
    results_dir = absolute_path(args.results_dir)
    output_dir = results_dir / "ml" / args.target

    if not args.dry_run:
        ensure_input_exists(input_path, "ML input file")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_ml_command(
        input_path=input_path,
        target=args.target,
        output_dir=output_dir,
        args=args,
    )

    expected_summary = output_dir / "results_summary.csv"

    run_command(
        command=command,
        step_name=f"Machine-learning prediction: {args.target}",
        expected_outputs=[expected_summary],
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\nML outputs:")
    print(f"  Output directory: {output_dir}")
    print(f"  Results summary:  {expected_summary}")
    print(f"  Report:           {output_dir / 'report.md'}")


def run_group_kfold(args: argparse.Namespace) -> None:
    """Run group_kfold_predictions.py on an existing imputed dataset."""
    ensure_script_exists(GROUP_KFOLD_SCRIPT)

    input_path = absolute_path(args.input)
    results_dir = absolute_path(args.results_dir)
    output_dir = results_dir / "group_kfold" / args.target

    if not args.dry_run:
        ensure_input_exists(input_path, "GroupKFold input file")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_group_kfold_command(
        input_path=input_path,
        target=args.target,
        output_dir=output_dir,
        args=args,
    )

    expected_summary = output_dir / "results_summary.csv"
    expected_predictions = output_dir / "predictions.csv"

    run_command(
        command=command,
        step_name=f"GroupKFold prediction: {args.target}",
        expected_outputs=[
            expected_summary,
            expected_predictions,
        ],
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\nGroupKFold outputs:")
    print(f"  Output directory: {output_dir}")
    print(f"  Results summary:  {expected_summary}")
    print(f"  Predictions:      {expected_predictions}")
    print(f"  Report:           {output_dir / 'report.md'}")    


def run_location_validation(args: argparse.Namespace) -> None:
    """Run reciprocal location validation on an imputed dataset."""
    ensure_script_exists(LOCATION_VALIDATION_SCRIPT)

    input_path = absolute_path(args.input)
    results_dir = absolute_path(args.results_dir)
    output_dir = results_dir / "location_validation" / args.target

    if not args.dry_run:
        ensure_input_exists(input_path, "Location-validation input file")

    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_location_validation_command(
        input_path=input_path,
        target=args.target,
        output_dir=output_dir,
        args=args,
    )

    expected_summary = output_dir / "results_summary.csv"

    run_command(
        command=command,
        step_name=f"Reciprocal location validation: {args.target}",
        expected_outputs=[expected_summary],
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\nLocation-validation outputs:")
    print(f"  Output directory: {output_dir}")
    print(f"  Results summary:  {expected_summary}")
    print(
        f"  Report:           "
        f"{output_dir / 'location_validation_report.md'}"
    )


def run_selection_efficiency(args: argparse.Namespace) -> None:
    """Run selection-efficiency analysis from GroupKFold predictions."""
    ensure_script_exists(SELECTION_EFFICIENCY_SCRIPT)

    results_dir = absolute_path(args.results_dir)

    if args.predictions:
        predictions_path = absolute_path(args.predictions)
        target_label = args.target or predictions_path.parent.name
    else:
        if not args.target:
            raise ValueError(
                "Provide either --predictions or --target."
            )

        target_label = args.target
        predictions_path = (
            results_dir
            / "group_kfold"
            / target_label
            / "predictions.csv"
        )

    if not args.dry_run:
        ensure_input_exists(
            predictions_path,
            "GroupKFold predictions file",
        )

    output_dir = (
        results_dir
        / "selection_efficiency"
        / target_label
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    command = build_selection_efficiency_command(
        predictions_path=predictions_path,
        output_dir=output_dir,
        args=args,
    )

    expected_summary = output_dir / "selection_summary.csv"

    run_command(
        command=command,
        step_name=f"Selection-efficiency analysis: {target_label}",
        expected_outputs=[expected_summary],
        force=args.force,
        dry_run=args.dry_run,
    )

    print("\nSelection-efficiency outputs:")
    print(f"  Predictions input: {predictions_path}")
    print(f"  Output directory:  {output_dir}")
    print(f"  Selection summary: {expected_summary}")
    print(f"  Report:            {output_dir / 'selection_report.md'}")


def run_analysis(args: argparse.Namespace) -> None:
    """Run selected prediction and post-processing analyses."""

    results_dir = absolute_path(args.results_dir)

    resolved_input = resolve_prediction_input(
        input_value=args.input,
        results_dir=results_dir,
        allow_missing=args.dry_run,
    )

    analysis_args = argparse.Namespace(**vars(args))
    analysis_args.input = str(resolved_input)

    selected = list(dict.fromkeys(args.analyses))

    print("\nSelected analysis modules:")
    for name in selected:
        print(f"  - {name}")

    if "selection-efficiency" in selected and "group-kfold" not in selected:
        predictions_path = (
            results_dir
            / "group_kfold"
            / args.target
            / "predictions.csv"
        )

        if not predictions_path.is_file() and not args.dry_run:
            raise ValueError(
                "selection-efficiency requires GroupKFold predictions. "
                "Add 'group-kfold' to --analyses or provide an existing "
                f"predictions file at: {predictions_path}"
            )

    if "ml" in selected:
        run_ml(analysis_args)

    if "group-kfold" in selected:
        run_group_kfold(analysis_args)

    if "location-validation" in selected:
        run_location_validation(analysis_args)

    if "selection-efficiency" in selected:
        selection_args = argparse.Namespace(**vars(analysis_args))
        selection_args.predictions = None
        run_selection_efficiency(selection_args)


def run_full(args: argparse.Namespace) -> None:
    """Run preprocessing followed by selected analyses."""

    print("\nStarting full PHENO_MaizE workflow.")

    # 1. Preprocessing
    run_preprocess(args)

    # 2. Resolve the imputed output generated by preprocessing
    results_dir = absolute_path(args.results_dir)
    extracted_dir = results_dir / "extracted"
    imputed_dir = results_dir / "imputed"

    wide_path = extracted_dir / args.output
    method_label = args.method.lower()

    imputed_path = (
        imputed_dir
        / f"{wide_path.stem}_imputed_{method_label}.csv"
    )

    if not args.dry_run:
        ensure_input_exists(
            imputed_path,
            "Imputed dataset generated by preprocessing",
        )

    # 3. Run analyses using the generated imputed dataset
    analysis_args = argparse.Namespace(**vars(args))
    analysis_args.input = str(imputed_path)

    run_analysis(analysis_args)


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by all subcommands."""
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the step again even when its output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Display commands without executing them.",
    )


def add_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    """Add extract.py-related arguments."""
    parser.add_argument(
        "--data-dir",
        default="example_data",
        help="Dataset directory (default: example_data).",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root pipeline output directory (default: results).",
    )
    parser.add_argument(
        "--flight-info",
        default="flight_info.csv",
        help="Flight metadata filename.",
    )
    parser.add_argument(
        "--field-data",
        default="field_data.csv",
        help="Field data filename.",
    )
    parser.add_argument(
        "--output",
        default="vi_results_wide.csv",
        help="Extracted wide-table filename.",
    )
    parser.add_argument(
        "--intermediate",
        default="vi_results_long.csv",
        help="Extracted long-table filename.",
    )


def add_imputation_arguments(
    parser: argparse.ArgumentParser,
    include_input: bool,
    include_custom_outputs: bool,
) -> None:
    """Add impute.py-related arguments."""
    if include_input:
        parser.add_argument(
            "--input",
            required=True,
            help="Input CSV containing missing feature values.",
        )
        parser.add_argument(
            "--results-dir",
            default="results",
            help="Root pipeline output directory (default: results).",
        )

    parser.add_argument(
        "--method",
        choices=["MICE", "KNN"],
        default="MICE",
        help="Imputation method (default: MICE).",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=None,
        help="Columns used for sequential grouped imputation.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=10,
        help="Maximum MICE iterations (default: 10).",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of KNN neighbors (default: 5).",
    )
    parser.add_argument(
        "--sep",
        default=None,
        help="CSV separator; otherwise impute.py auto-detects it.",
    )

    if include_custom_outputs:
        parser.add_argument(
            "--output",
            default=None,
            help="Optional custom path for the imputed CSV.",
        )
        parser.add_argument(
            "--flag-output",
            default=None,
            help="Optional custom path for the imputation flags CSV.",
        )


def add_prediction_arguments(
    parser: argparse.ArgumentParser,
    require_input: bool = True,
) -> None:
    """Add arguments shared by prediction scripts."""

    parser.add_argument(
        "--input",
        required=require_input,
        default=None,
        help=(
            "Input CSV with imputed features. For the analysis command, "
            "the file is resolved automatically when omitted."
        ),
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Numeric response column to predict.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root pipeline output directory.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["Lasso", "Ridge", "RF", "XGBoost", "CNN"],
        default=None,
        help="Models to run.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Regularization parameter for Lasso and Ridge.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of estimators for tree-based models.",
    )
    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=None,
        help="Number of CNN training epochs.",
    )
    parser.add_argument(
        "--feature-set",
        choices=["all", "veg_qgis", "veg_exg80", "combined"],
        default=None,
        help="Feature subset based on vegetation-mask suffixes.",
    )
    parser.add_argument(
        "--stats",
        choices=["mean", "std", "mean_std"],
        default=None,
        help="Statistics retained from vegetation-index columns.",
    )
    parser.add_argument(
        "--include-canopy",
        action="store_true",
        help="Include canopy-cover columns.",
    )
    parser.add_argument(
        "--include-regex",
        default=None,
        help="Keep only features matching this regex.",
    )
    parser.add_argument(
        "--exclude-regex",
        default=None,
        help="Remove features matching this regex.",
    )
    parser.add_argument(
        "--vegetation-indices",
        nargs="+",
        default=None,
        help="Exact vegetation-index names to retain.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Growth-stage names to retain.",
    )
    parser.add_argument(
        "--show-feature-summary",
        action="store_true",
        help="Print a summary of selected features.",
    )


def add_group_kfold_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add arguments specific to GroupKFold validation."""

    parser.add_argument(
        "--cv-mode",
        choices=["random", "hybrid_groupkfold"],
        default="hybrid_groupkfold",
        help="Validation mode.",
    )
    parser.add_argument(
        "--group-col",
        default="genotype",
        help="Column identifying genotypes that must remain in the same fold.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of folds for GroupKFold.",
    )


def add_selection_efficiency_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add arguments for selection-efficiency analysis."""

    parser.add_argument(
        "--predictions",
        default=None,
        help=(
            "Path to predictions.csv. If omitted, --target is used to "
            "resolve results/group_kfold/<target>/predictions.csv."
        ),
    )
    parser.add_argument(
        "--target",
        default=None,
        help="Target label used to resolve the default predictions path.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Root pipeline output directory.",
    )
    parser.add_argument(
        "--group-col",
        default="genotype",
        help="Genotype/group identifier column in predictions.csv.",
    )
    parser.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Top selection fractions.",
    )
    parser.add_argument(
        "--bottom-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Bottom selection fractions.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional model names to include.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional experiment names to include.",
    )


def add_analysis_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add arguments for the combined prediction-analysis workflow."""

    add_prediction_arguments(parser, require_input=False)

    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=[
            "ml",
            "group-kfold",
            "location-validation",
            "selection-efficiency",
        ],
        default=[
            "ml",
            "group-kfold",
            "location-validation",
            "selection-efficiency",
        ],
        help=(
            "Analysis modules to run. By default, all prediction and "
            "selection-efficiency modules are executed."
        ),
    )

    parser.add_argument(
        "--cv-mode",
        choices=["random", "hybrid_groupkfold"],
        default="hybrid_groupkfold",
        help="Validation mode used by the GroupKFold module.",
    )

    parser.add_argument(
        "--group-col",
        default="genotype",
        help=(
            "Column identifying genotypes that must remain together "
            "within GroupKFold splits."
        ),
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits.",
    )

    parser.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Top-selection fractions for selection-efficiency analysis.",
    )

    parser.add_argument(
        "--bottom-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Bottom-selection fractions for selection-efficiency analysis.",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help=(
            "Optional experiment names retained in selection-efficiency "
            "analysis."
        ),
    )


def add_full_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add arguments for the complete PHENO_MaizE workflow."""

    add_extraction_arguments(parser)

    add_imputation_arguments(
        parser,
        include_input=False,
        include_custom_outputs=False,
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Numeric response column to predict.",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["Lasso", "Ridge", "RF", "XGBoost", "CNN"],
        default=None,
        help="Models to run.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Regularization parameter for Lasso and Ridge.",
    )

    parser.add_argument(
        "--n-estimators",
        type=int,
        default=None,
        help="Number of estimators for tree-based models.",
    )

    parser.add_argument(
        "--cnn-epochs",
        type=int,
        default=None,
        help="Number of CNN training epochs.",
    )

    parser.add_argument(
        "--feature-set",
        choices=["all", "veg_qgis", "veg_exg80", "combined"],
        default=None,
        help="Feature subset based on vegetation-mask suffixes.",
    )

    parser.add_argument(
        "--stats",
        choices=["mean", "std", "mean_std"],
        default=None,
        help="Statistics retained from vegetation-index columns.",
    )

    parser.add_argument(
        "--include-canopy",
        action="store_true",
        help="Include canopy-cover columns.",
    )

    parser.add_argument(
        "--include-regex",
        default=None,
        help="Keep only features matching this regex.",
    )

    parser.add_argument(
        "--exclude-regex",
        default=None,
        help="Remove features matching this regex.",
    )

    parser.add_argument(
        "--vegetation-indices",
        nargs="+",
        default=None,
        help="Exact vegetation-index names to retain.",
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help="Growth-stage names to retain.",
    )

    parser.add_argument(
        "--show-feature-summary",
        action="store_true",
        help="Print a summary of selected features.",
    )

    parser.add_argument(
        "--analyses",
        nargs="+",
        choices=[
            "ml",
            "group-kfold",
            "location-validation",
            "selection-efficiency",
        ],
        default=[
            "ml",
            "group-kfold",
            "location-validation",
            "selection-efficiency",
        ],
        help="Analysis modules to run after preprocessing.",
    )

    parser.add_argument(
        "--cv-mode",
        choices=["random", "hybrid_groupkfold"],
        default="hybrid_groupkfold",
        help="Validation mode used by the GroupKFold module.",
    )

    parser.add_argument(
        "--group-col",
        default="genotype",
        help="Column identifying genotypes that must remain in the same fold.",
    )

    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold splits.",
    )

    parser.add_argument(
        "--top-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Top-selection fractions.",
    )

    parser.add_argument(
        "--bottom-fractions",
        nargs="+",
        type=float,
        default=None,
        help="Bottom-selection fractions.",
    )

    parser.add_argument(
        "--experiments",
        nargs="+",
        default=None,
        help="Optional experiment names for selection-efficiency analysis.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run PHENO_MaizE pipeline modules."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    extract_parser = subparsers.add_parser(
        "extract",
        help="Run vegetation-index extraction.",
    )
    add_extraction_arguments(extract_parser)
    add_execution_arguments(extract_parser)
    extract_parser.set_defaults(function=run_extract)

    impute_parser = subparsers.add_parser(
        "impute",
        help="Run missing-value imputation on an existing CSV.",
    )
    add_imputation_arguments(
        impute_parser,
        include_input=True,
        include_custom_outputs=True,
    )
    add_execution_arguments(impute_parser)
    impute_parser.set_defaults(function=run_impute)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Run extraction followed by imputation.",
    )
    add_extraction_arguments(preprocess_parser)
    add_imputation_arguments(
        preprocess_parser,
        include_input=False,
        include_custom_outputs=False,
    )
    add_execution_arguments(preprocess_parser)
    preprocess_parser.set_defaults(function=run_preprocess)
    
    ml_parser = subparsers.add_parser(
        "ml",
        help="Run standard machine-learning prediction.",
    )
    add_prediction_arguments(ml_parser)
    add_execution_arguments(ml_parser)
    ml_parser.set_defaults(function=run_ml)

    group_kfold_parser = subparsers.add_parser(
        "group-kfold",
        help="Run GroupKFold machine-learning validation.",
    )
    add_prediction_arguments(group_kfold_parser)
    add_group_kfold_arguments(group_kfold_parser)
    add_execution_arguments(group_kfold_parser)
    group_kfold_parser.set_defaults(function=run_group_kfold)

    location_validation_parser = subparsers.add_parser(
        "location-validation",
        help="Run reciprocal location validation.",
    )
    add_prediction_arguments(location_validation_parser)
    add_execution_arguments(location_validation_parser)
    location_validation_parser.set_defaults(
        function=run_location_validation
    )

    analysis_parser = subparsers.add_parser(
        "analysis",
        help="Run prediction analyses and post-processing.",
    )
    add_analysis_arguments(analysis_parser)
    add_execution_arguments(analysis_parser)
    analysis_parser.set_defaults(function=run_analysis)

    selection_efficiency_parser = subparsers.add_parser(
        "selection-efficiency",
        help="Run genotype selection-efficiency analysis.",
    )
    add_selection_efficiency_arguments(selection_efficiency_parser)
    add_execution_arguments(selection_efficiency_parser)
    selection_efficiency_parser.set_defaults(
        function=run_selection_efficiency
    )

    full_parser = subparsers.add_parser(
        "full",
        help="Run preprocessing followed by selected analyses.",
    )
    add_full_arguments(full_parser)
    add_execution_arguments(full_parser)
    full_parser.set_defaults(function=run_full)
    
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        args.function(args)
    except subprocess.CalledProcessError as exc:
        print(
            f"\n[ERROR] Pipeline command failed with exit code "
            f"{exc.returncode}.",
            file=sys.stderr,
        )
        return exc.returncode or 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n[STOP] Pipeline interrupted by user.", file=sys.stderr)
        return 130

    print("\nPipeline step completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())