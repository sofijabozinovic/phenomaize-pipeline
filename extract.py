#!/usr/bin/env python3
"""
PhenoMaize VI Extraction Module (RGB) - dual vegetation masks (QGIS vs ExG)

What it does (per plot per orthomosaic):
- Computes vegetation indices (VIs) from RGB (same VI set as before)
- Computes stats for each VI in THREE variants:
  (1) ALL:  mean/std over ALL valid pixels (veg + soil + shadows)
  (2) VEG_QGIS: mean/std over QGIS protocol vegetation mask (GLI + HSV(S,V) + 3x3 majority)
  (3) VEG_EXG80: mean/std over ExG mask with fixed threshold ExG > 80 (+ brightness filter)
- Computes canopy cover for BOTH veg masks:
  - canopy_cover_qgis
  - canopy_cover_exg80

Outputs:
1) intermediate long CSV (one row per plot x orthomosaic)
2) final wide CSV pivoted by Growth stage (stage_prefix)

Assumptions:
- Orthomosaic is RGB in bands 1..3 (R,G,B) with values 0–255 (or similar scale).
- One shapefile named "plots.shp" exists for each trial:
    shapefiles/<trial>/plots.shp
- Plot identifiers are stored in the "PlotID" attribute.
- Flight metadata are read from flight_info.csv.
"""

import os
import sys
import argparse
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import numpy as np
from tqdm.auto import tqdm
import warnings
import re
import gc
from multiprocessing import Pool, cpu_count


# ----------------------------
# Mask settings
# ----------------------------
BRIGHTNESS_MIN = 20  # common shadow filter, used by both masks

# ExG mask (fixed)
EXG_THRESHOLD = 80  # only exg80 requested

# QGIS protocol mask parameters
QGIS_GLI_THR = 0.18
QGIS_S_THR = 0.22
QGIS_V_THR = 65
QGIS_APPLY_MODE3 = True


# ----------------------------
# QGIS protocol mask helpers (NumPy-only)
# ----------------------------
def _majority_filter_3x3(binary01: np.ndarray) -> np.ndarray:
    """
    Majority vote 3x3 over a binary mask (0/1). NumPy-only (no SciPy).
    """
    p = np.pad(binary01.astype(np.uint8), ((1, 1), (1, 1)), mode="edge")
    s = (
        p[0:-2, 0:-2] + p[0:-2, 1:-1] + p[0:-2, 2:] +
        p[1:-1, 0:-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
        p[2:,   0:-2] + p[2:,   1:-1] + p[2:,   2:]
    )
    return (s >= 5).astype(np.uint8)


def qgis_protocol_mask(R: np.ndarray, G: np.ndarray, B: np.ndarray,
                       gli_thr: float = QGIS_GLI_THR,
                       s_thr: float = QGIS_S_THR,
                       v_thr: float = QGIS_V_THR,
                       apply_mode3: bool = QGIS_APPLY_MODE3) -> np.ndarray:
    """
    Boolean vegetation mask by QGIS protocol:
      V = max(R,G,B)
      S = (V - min(R,G,B)) / V  (safe for V>0)
      GLI = (2G - R - B)/(2G + R + B)
      veg = (GLI > gli_thr) & (S > s_thr) & (V > v_thr)
      optional majority filter 3x3

    Input expects 2D arrays. If data are >255 (e.g., 16-bit), it rescales to 0..255.
    """
    Rf = R.astype(np.float32)
    Gf = G.astype(np.float32)
    Bf = B.astype(np.float32)

    maxv = np.nanmax([Rf.max(), Gf.max(), Bf.max()])
    if maxv > 255.0:
        scale = 255.0 / maxv
        Rf *= scale
        Gf *= scale
        Bf *= scale

    V = np.maximum.reduce([Rf, Gf, Bf])
    min_rgb = np.minimum.reduce([Rf, Gf, Bf])

    S = np.zeros_like(V, dtype=np.float32)
    m = V > 0
    S[m] = (V[m] - min_rgb[m]) / V[m]  # 0..1

    den = 2 * Gf + Rf + Bf
    GLI = np.full_like(Gf, np.nan, dtype=np.float32)
    md = den > 0
    GLI[md] = (2 * Gf[md] - Rf[md] - Bf[md]) / den[md]

    veg = (GLI > gli_thr) & (S > s_thr) & (V > float(v_thr))

    if apply_mode3:
        veg01 = _majority_filter_3x3(veg.astype(np.uint8))
        return (veg01 == 1)
    return veg


# ----------------------------
# Vegetation indices
# ----------------------------
def calculate_vis(r, g, b):
    """
    Calculate vegetation indices from RGB bands.
    Inputs: r,g,b are 1D arrays (float32), typically 0–255.
    Output: dict of VI arrays (same length).
    """
    eps = 1e-8
    vi = {}

    r = r.astype(np.float32)
    g = g.astype(np.float32)
    b = b.astype(np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        # VDVI
        denom = 2 * g + r + b
        vi["VDVI"] = np.divide((2 * g - r - b), denom, where=denom != 0,
                               out=np.full_like(denom, np.nan, dtype=np.float32))

        # NGRDI
        denom = g + r
        vi["NGRDI"] = np.divide((g - r), denom, where=denom != 0,
                                out=np.full_like(denom, np.nan, dtype=np.float32))

        # VARI
        denom = g + r - b
        vi["VARI"] = np.divide((g - r), denom, where=denom != 0,
                               out=np.full_like(denom, np.nan, dtype=np.float32))

        # GRRI
        denom = r
        vi["GRRI"] = np.divide(g, denom, where=denom != 0,
                               out=np.full_like(denom, np.nan, dtype=np.float32))

        # VEG
        a = 0.667
        r_pow_a = np.power(np.maximum(r, eps), a)
        b_pow_1ma = np.power(np.maximum(b, eps), (1 - a))
        denom = r_pow_a * b_pow_1ma
        vi["VEG"] = np.divide(g, denom, where=denom != 0,
                              out=np.full_like(denom, np.nan, dtype=np.float32))

        # MGRVI
        g2 = g * g
        r2 = r * r
        denom = g2 + r2
        vi["MGRVI"] = np.divide((g2 - r2), denom, where=denom != 0,
                                out=np.full_like(denom, np.nan, dtype=np.float32))

        # ExG
        vi["ExG"] = 2 * g - r - b

        # RGBVI
        denom = (g * g) + (r * b)
        vi["RGBVI"] = np.divide((g * g - r * b), denom, where=denom != 0,
                                out=np.full_like(denom, np.nan, dtype=np.float32))

        # TGI
        vi["TGI"] = -0.5 * (190 * (r - g) - 120 * (r - b))

        # ExGR
        vi["ExGR"] = 3.0 * g - 2.4 * r - b

        # CIVE
        vi["CIVE"] = 0.441 * r - 0.811 * g + 0.385 * b

        # MExG
        vi["MExG"] = 1.262 * g - 0.884 * r - 0.311 * b

        # BGI
        denom = g
        vi["BGI"] = np.divide(b, denom, where=denom != 0,
                              out=np.full_like(denom, np.nan, dtype=np.float32))

        # BI
        vi["BI"] = np.sqrt((r * r + g * g + b * b) / 3.0)

        # GLI
        denom = 2 * g + r + b
        vi["GLI"] = np.divide((2 * g - r - b), denom, where=denom != 0,
                              out=np.full_like(denom, np.nan, dtype=np.float32))

        # NDI
        denom = g + r
        ngrdi = np.divide((g - r), denom, where=denom != 0,
                          out=np.full_like(denom, np.nan, dtype=np.float32))
        vi["NDI"] = 128.0 * (ngrdi + 1.0)

        # NGBDI
        denom = g + b
        vi["NGBDI"] = np.divide((g - b), denom, where=denom != 0,
                                out=np.full_like(denom, np.nan, dtype=np.float32))

    # Clean infinities
    for k, arr in vi.items():
        if isinstance(arr, np.ndarray):
            arr[np.isinf(arr)] = np.nan

    return vi


# Pre-compute VI names for consistent NaN-filling in empty masks
VI_NAMES = list(calculate_vis(np.array([1], dtype=np.float32),
                              np.array([1], dtype=np.float32),
                              np.array([1], dtype=np.float32)).keys())


def sanitize_column_name(name):
    """Sanitize column names for safe CSV usage."""
    name = str(name).strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[(),./-]", "", name)
    return name


def compute_vi_stats(r, g, b, suffix, stats=("mean", "std")):
    """
    Compute VI stats for given 1D RGB arrays.
    Returns dict with keys like: VDVI_mean_all / VDVI_std_all (suffix appended).
    """
    vi = calculate_vis(r, g, b)
    stats_fns = {"mean": np.nanmean, "std": np.nanstd}
    out = {}
    for vi_name, vals in vi.items():
        all_nan = np.all(np.isnan(vals))
        for s in stats:
            out[f"{vi_name}_{s}_{suffix}"] = np.nan if all_nan else float(stats_fns[s](vals))
    return out


def process_orthomosaic(flight_row_tuple):
    """Extract VI statistics and canopy cover for all plots in one orthomosaic."""
    _, flight_row = flight_row_tuple
    ortho_path = flight_row["FilePath"]
    current_trial = flight_row["trial"]
    current_date = flight_row["date"]
    current_growth_stage = flight_row["growth_stage"]
    ortho_filename = os.path.basename(ortho_path)
    tif_results = []

    # Load the shapefile associated with the current trial
    data_dir = os.path.dirname(os.path.dirname(os.path.dirname(ortho_path)))

    shp_path = os.path.join(
        data_dir,
        "shapefiles",
        current_trial,
        "plots.shp"
    )

    if not os.path.exists(shp_path):
        print(f"Shapefile not found: {shp_path}")
        return []

    try:
        current_polygons_gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"Error reading shapefile {shp_path}: {e}")
        return []

    if current_polygons_gdf.empty:
        return []

    if "PlotID" not in current_polygons_gdf.columns:
        print(f"Missing 'PlotID' column in {shp_path}")
        return []

    current_polygons_gdf = current_polygons_gdf[
        ["PlotID", current_polygons_gdf.geometry.name]
    ].copy()

    current_polygons_gdf.rename(
        columns={"PlotID": "plot_id"},
        inplace=True
    )

    current_polygons_gdf["plot_id"] = (
        current_polygons_gdf["plot_id"]
        .astype(str)
        .str.strip()
    )

    current_polygons_gdf["trial"] = current_trial

    current_polygons_gdf = current_polygons_gdf.dropna(
        subset=["plot_id"]
    )

    if current_polygons_gdf.empty:
        return []

    try:
        with rasterio.open(ortho_path) as src:
            if src.crs and current_polygons_gdf.crs and src.crs != current_polygons_gdf.crs:
                current_polygons_gdf = current_polygons_gdf.to_crs(src.crs)

            if src.count < 3:
                print(
                    f"Orthomosaic has fewer than 3 bands: "
                    f"{ortho_path} ({src.count} bands)"
                )
                return []

            nodata_val_meta = src.nodata
            nodata_fill_val = nodata_val_meta if nodata_val_meta is not None else 0

            for plot_geom in current_polygons_gdf.itertuples():
                try:
                    out_image, _ = rasterio.mask.mask(
                        src, [plot_geom.geometry],
                        crop=True, filled=True, nodata=nodata_fill_val
                    )
                    if out_image.size == 0 or out_image.shape[0] < 3:
                        continue

                    r2d = out_image[0, :, :]
                    g2d = out_image[1, :, :]
                    b2d = out_image[2, :, :]

                    # valid pixels
                    if nodata_val_meta is not None:
                        valid_mask = (r2d != nodata_val_meta) & (g2d != nodata_val_meta) & (b2d != nodata_val_meta)
                    else:
                        valid_mask = ~((r2d == 0) & (g2d == 0) & (b2d == 0))

                    total_valid = int(np.sum(valid_mask))
                    if total_valid == 0:
                        continue

                    # brightness filter
                    r_f = r2d.astype(np.float32)
                    g_f = g2d.astype(np.float32)
                    b_f = b2d.astype(np.float32)
                    brightness = (r_f + g_f + b_f) / 3.0
                    bright_mask = brightness > BRIGHTNESS_MIN

                    # --- ExG80 mask (fixed) ---
                    exg2d = 2.0 * g_f - r_f - b_f
                    veg_mask_exg80 = valid_mask & bright_mask & (exg2d > EXG_THRESHOLD)
                    canopy_cover_exg80 = 100.0 * float(np.sum(veg_mask_exg80)) / float(total_valid)

                    # --- QGIS protocol mask ---
                    veg_mask_qgis = valid_mask & bright_mask & qgis_protocol_mask(
                        r2d, g2d, b2d,
                        gli_thr=QGIS_GLI_THR, s_thr=QGIS_S_THR, v_thr=QGIS_V_THR,
                        apply_mode3=QGIS_APPLY_MODE3
                    )
                    canopy_cover_qgis = 100.0 * float(np.sum(veg_mask_qgis)) / float(total_valid)

                    # ---- VI stats: ALL valid pixels ----
                    r_all = r2d[valid_mask]
                    g_all = g2d[valid_mask]
                    b_all = b2d[valid_mask]
                    vi_stats_all = compute_vi_stats(r_all, g_all, b_all, suffix="all")

                    # ---- VI stats: VEG_EXG80 ----
                    if np.sum(veg_mask_exg80) > 0:
                        r_exg = r2d[veg_mask_exg80]
                        g_exg = g2d[veg_mask_exg80]
                        b_exg = b2d[veg_mask_exg80]
                        vi_stats_exg = compute_vi_stats(r_exg, g_exg, b_exg, suffix="veg_exg80")
                    else:
                        vi_stats_exg = {}
                        for n in VI_NAMES:
                            vi_stats_exg[f"{n}_mean_veg_exg80"] = np.nan
                            vi_stats_exg[f"{n}_std_veg_exg80"] = np.nan

                    # ---- VI stats: VEG_QGIS ----
                    if np.sum(veg_mask_qgis) > 0:
                        r_q = r2d[veg_mask_qgis]
                        g_q = g2d[veg_mask_qgis]
                        b_q = b2d[veg_mask_qgis]
                        vi_stats_qgis = compute_vi_stats(r_q, g_q, b_q, suffix="veg_qgis")
                    else:
                        vi_stats_qgis = {}
                        for n in VI_NAMES:
                            vi_stats_qgis[f"{n}_mean_veg_qgis"] = np.nan
                            vi_stats_qgis[f"{n}_std_veg_qgis"] = np.nan

                    # store row
                    plot_result = {
                        "plot_id": plot_geom.plot_id,
                        "trial": current_trial,
                        "date": current_date,
                        "growth_stage": current_growth_stage,
                        "orthomosaic": ortho_filename,
                        "canopy_cover_exg80": canopy_cover_exg80,
                        "canopy_cover_qgis": canopy_cover_qgis,
                    }
                    plot_result.update(vi_stats_all)
                    plot_result.update(vi_stats_exg)
                    plot_result.update(vi_stats_qgis)

                    tif_results.append(plot_result)

                except Exception as e:
                    print(
                        f"Error processing plot {plot_geom.plot_id} "
                        f"in orthomosaic {ortho_filename}: {e}"
                    )
                    

    except Exception as e:
        print(f"Error processing {ortho_path}: {e}")
        return []

    return tif_results


def main():
    parser = argparse.ArgumentParser(
        description="Extract VIs (RGB) with dual vegetation masks (QGIS vs ExG80)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
          python extract.py

          python extract.py --data-dir /path/to/my_dataset
        """
    )

    parser.add_argument(
        "--data-dir",
        default="example_data",
        help="Directory containing the input dataset (default: example_data)"
    )

    parser.add_argument(
        "--flight-info",
        default="flight_info.csv",
        help="Flight metadata CSV filename (default: flight_info.csv)"
    )

    parser.add_argument(
        "--field-data",
        default="field_data.csv",
        help="Field data CSV filename (default: field_data.csv)"
    )

    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for output files (default: results)"
    )

    parser.add_argument(
        "--output",
        default="vi_results_wide.csv",
        help="Final output filename (default: vi_results_wide.csv)"
    )

    parser.add_argument(
        "--intermediate",
        default="vi_results_long.csv",
        help="Intermediate output filename (default: vi_results_long.csv)"
    )

    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)

    flight_info_path = os.path.join(data_dir, args.flight_info)
    field_data_path = os.path.join(data_dir, args.field_data)

    # Create results directory if it doesn't exist
    results_dir = os.path.join(data_dir, args.results_dir)
    os.makedirs(results_dir, exist_ok=True)

    output_path = os.path.join(results_dir, args.output)
    intermediate_path = os.path.join(results_dir, args.intermediate)

    print("=" * 70)
    print("PhenoMaize VI Extraction (ALL + VEG_EXG80 + VEG_QGIS)")
    print("=" * 70)
    print(f"\nData directory: {data_dir}")
    print(f"Flight info: {flight_info_path}")
    print(f"Field data: {field_data_path}")
    print()
    print("Masks:")
    print(f"  EXG80: ExG > {EXG_THRESHOLD} AND brightness > {BRIGHTNESS_MIN}")
    print(f"  QGIS : GLI>{QGIS_GLI_THR}, S>{QGIS_S_THR}, V>{QGIS_V_THR}, mode3={QGIS_APPLY_MODE3}, AND brightness>{BRIGHTNESS_MIN}\n")

    # Validate inputs
    if not os.path.exists(flight_info_path):
        print(f"ERROR: Flight info file not found: {flight_info_path}")
        sys.exit(1)
    if not os.path.exists(field_data_path):
        print(f"ERROR: Field data file not found: {field_data_path}")
        sys.exit(1)

    # Load data
    print("Loading input data...")
    flight_info_df = pd.read_csv(flight_info_path, skipinitialspace=True, sep=",")
    data_df = pd.read_csv(field_data_path, skipinitialspace=True, sep=",")

    # Prepare flight info
    flight_info_df = flight_info_df.dropna(
        subset=["location", "trial", "date", "growth_stage", "file_name"]
    ).copy()

    flight_info_df["location"] = (
        flight_info_df["location"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    flight_info_df["trial"] = (
        flight_info_df["trial"]
        .astype(str)
        .str.strip()
    )

    flight_info_df["file_name"] = (
        flight_info_df["file_name"]
        .astype(str)
        .str.strip()
    )

    flight_info_df["growth_stage"] = (
        flight_info_df["growth_stage"]
        .astype(str)
        .str.strip()
    )

    flight_info_df["date"] = pd.to_datetime(
        flight_info_df["date"].astype(str).str.rstrip("."),
        format="%d.%m.%y",
        errors="coerce"
    )

    flight_info_df = flight_info_df.dropna(subset=["date"])
    flight_info_df = flight_info_df.sort_values(
        by=["location", "trial", "date"]
    ).reset_index(drop=True)

    # Build orthomosaic file paths
    flight_info_df["FilePath"] = flight_info_df.apply(
    lambda row: os.path.join(
        data_dir,
        "orthomosaics",
        str(row["trial"]).strip(),
        str(row["file_name"]).strip()
    ),
    axis=1
    )

    missing_files = flight_info_df[
        ~flight_info_df["FilePath"].apply(os.path.exists)
    ]

    if not missing_files.empty:
        print("\nWarning: The following orthomosaics were not found:")
        for _, row in missing_files.iterrows():
            print(f"  {row['FilePath']}")

    flight_info_df = flight_info_df[
        flight_info_df["FilePath"].apply(os.path.exists)
    ].copy()

    flight_info_df = flight_info_df.sort_values(
        by=["location", "trial", "date"]
    ).reset_index(drop=True)

    # Prepare field data
    data_df["trial"] = data_df["trial"].astype(str).str.strip()
    data_df["plot_id"] = data_df["plot_id"].astype(str).str.strip()

    print(f"Found {len(flight_info_df)} flights to process")
    print(f"Field data contains {len(data_df)} plots")

    # Parallel processing
    num_processes = max(1, cpu_count() - 1)
    print(f"\nProcessing orthomosaics using {num_processes} CPU cores...")

    all_results = []
    with Pool(processes=num_processes) as pool:
        pbar = tqdm(
            pool.imap_unordered(process_orthomosaic, flight_info_df.iterrows()),
            total=len(flight_info_df),
            desc="Extracting VIs"
        )
        for tif_results in pbar:
            if tif_results:
                all_results.extend(tif_results)

    print(f"\nExtracted data for {len(all_results)} plot-flight combinations")
    if not all_results:
        print("\nWARNING: No VI results generated. Check input data and file paths.")
        sys.exit(1)

    # Save intermediate results in long format
    print("\nSaving intermediate results (long format)...")

    vi_results_df = pd.DataFrame(all_results)

    vi_results_df.to_csv(
        intermediate_path,
        index=False,
        encoding="utf-8-sig"
    )

    print(f"  Saved: {intermediate_path}")


    # Pivot results to wide format by growth stage
    print("\nPivoting data to wide format...")

    vi_results_df["stage_prefix"] = (
        vi_results_df["growth_stage"]
        .astype(str)
        .apply(sanitize_column_name)
    )

    stats = ["mean", "std"]
    suffixes = ["all", "veg_exg80", "veg_qgis"]

    value_cols = []

    for vi_name in VI_NAMES:
        for statistic in stats:
            for suffix in suffixes:
                value_cols.append(
                    f"{vi_name}_{statistic}_{suffix}"
                )

    value_cols.extend([
        "canopy_cover_exg80",
        "canopy_cover_qgis"
    ])

    vi_pivot_df = vi_results_df.pivot_table(
        index=["plot_id", "trial"],
        columns="stage_prefix",
        values=value_cols,
        aggfunc="first"
    )

    # Flatten MultiIndex columns:
    # ('VDVI_mean_all', 'veg_mid_1') -> 'veg_mid_1_VDVI_mean_all'
    vi_pivot_df.columns = [f"{stage}_{col}" for col, stage in vi_pivot_df.columns]
    vi_pivot_df.reset_index(inplace=True)

    # Merge with field data
    print("Merging with field data...")
    final_df = pd.merge(
        data_df,
        vi_pivot_df,
        on=["plot_id", "trial"],
        how="left"
    )
    print(f"Final dataset shape: {final_df.shape}")

    # Save final (wide)
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSuccessfully saved final results to: {output_path}")

    # Cleanup
    del flight_info_df, data_df, all_results, vi_results_df, vi_pivot_df, final_df
    gc.collect()

    print("\n" + "=" * 70)
    print("Extraction complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
