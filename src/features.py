from typing import List
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    feature_config = config["features"]["engineering"]

    period_cols = [c for c in df.columns if "period" in c.lower()]
    duration_cols = [c for c in df.columns if "duration" in c.lower()]
    radius_cols = [c for c in df.columns if "prad" in c.lower() or "radius" in c.lower()]
    depth_cols = [c for c in df.columns if "depth" in c.lower()]

    if feature_config.get("log_transform"):
        for col_pattern in feature_config["log_transform"]:
            matching_cols = [c for c in df.columns if col_pattern in c]
            for col in matching_cols:
                if col in df.columns and df[col].dtype in [np.float64, np.int64]:
                    df[f"{col}_log"] = np.log1p(df[col].clip(lower=0))

    if feature_config.get("ratios") and period_cols and duration_cols:
        for p_col in period_cols[:1]:
            for d_col in duration_cols[:1]:
                if p_col in df.columns and d_col in df.columns:
                    df["period_duration_ratio"] = df[p_col] / (df[d_col] + 1e-9)

    if radius_cols:
        for r_col in radius_cols[:1]:
            if r_col in df.columns:
                df[f"{r_col}_squared"] = df[r_col] ** 2

    if depth_cols and duration_cols:
        for depth_col in depth_cols[:1]:
            for dur_col in duration_cols[:1]:
                if depth_col in df.columns and dur_col in df.columns:
                    df["depth_duration_product"] = df[depth_col] * df[dur_col]

    snr_cols = [c for c in df.columns if "snr" in c.lower()]
    if not snr_cols and depth_cols:
        for depth_col in depth_cols[:1]:
            if depth_col in df.columns:
                df["estimated_snr"] = np.sqrt(np.abs(df[depth_col]) + 1e-9)

    return df


def select_features(df: pd.DataFrame, config: dict) -> List[str]:
    exclude_patterns = [
        "id", "name", "disposition", "disp", "comment", "flag",
        "url", "date", "mission", "target", "tce", "kepid", "tic"
    ]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    selected = []
    for col in numeric_cols:
        if not any(pattern in col.lower() for pattern in exclude_patterns):
            missing_ratio = df[col].isna().sum() / len(df)
            if missing_ratio < config["preprocessing"]["missing_threshold"]:
                selected.append(col)

    return selected
