from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import yaml


def load_config() -> Dict:
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def map_disposition(value: str, config: Dict) -> str:
    if pd.isna(value):
        return "unknown"

    value_upper = str(value).upper().strip()

    if any(conf in value_upper for conf in config["labels"]["confirmed"]):
        return "confirmed"
    elif any(cand in value_upper for cand in config["labels"]["candidate"]):
        return "candidate"
    elif any(fp in value_upper for fp in config["labels"]["false_positive"]):
        return "false_positive"
    else:
        return "unknown"


def load_koi_data(config: Dict) -> pd.DataFrame:
    raw_dir = Path(config["data"]["raw_dir"])
    df = pd.read_csv(raw_dir / "koi.csv", low_memory=False)

    disp_col = config["labels"]["koi_disposition_col"]
    if disp_col in df.columns:
        df["target"] = df[disp_col].apply(lambda x: map_disposition(x, config))
    else:
        df["target"] = "unknown"

    df["mission"] = "kepler"
    return df


def load_toi_data(config: Dict) -> pd.DataFrame:
    raw_dir = Path(config["data"]["raw_dir"])
    df = pd.read_csv(raw_dir / "toi.csv", low_memory=False)

    disp_col = config["labels"]["toi_disposition_col"]
    if disp_col in df.columns:
        df["target"] = df[disp_col].apply(lambda x: map_disposition(x, config))
    else:
        df["target"] = "unknown"

    df["mission"] = "tess"
    return df


def load_k2_data(config: Dict) -> pd.DataFrame:
    raw_dir = Path(config["data"]["raw_dir"])
    df = pd.read_csv(raw_dir / "k2.csv", low_memory=False)

    disp_col = config["labels"]["k2_disposition_col"]
    if disp_col in df.columns:
        df["target"] = df[disp_col].apply(lambda x: map_disposition(x, config))
    else:
        df["target"] = "unknown"

    df["mission"] = "k2"
    return df


def load_all_data(config: Dict, filter_unknown: bool = True) -> pd.DataFrame:
    dfs = []

    try:
        koi = load_koi_data(config)
        print(f"Loaded KOI: {len(koi)} rows")
        dfs.append(koi)
    except Exception as e:
        print(f"Could not load KOI: {e}")

    try:
        toi = load_toi_data(config)
        print(f"Loaded TOI: {len(toi)} rows")
        dfs.append(toi)
    except Exception as e:
        print(f"Could not load TOI: {e}")

    try:
        k2 = load_k2_data(config)
        print(f"Loaded K2: {len(k2)} rows")
        dfs.append(k2)
    except Exception as e:
        print(f"Could not load K2: {e}")

    if not dfs:
        raise ValueError("No datasets could be loaded")

    df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    if filter_unknown:
        df = df[df["target"] != "unknown"]

    print(f"\nCombined dataset: {len(df)} rows")
    print(f"Label distribution:\n{df['target'].value_counts()}")

    return df
