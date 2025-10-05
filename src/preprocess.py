from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def create_preprocessing_pipeline(numeric_features: List[str], config: dict) -> ColumnTransformer:
    scaler_type = config["preprocessing"].get("scaler", "robust")

    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features)
        ],
        remainder="drop"
    )

    return preprocessor


def prepare_data(
        df: pd.DataFrame,
        feature_cols: List[str],
        config: dict
) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    df_clean = df[feature_cols + ["target"]].copy()
    df_clean = df_clean.dropna(subset=["target"])

    le = LabelEncoder()
    y = le.fit_transform(df_clean["target"])

    X = df_clean[feature_cols]

    return X, y, le
