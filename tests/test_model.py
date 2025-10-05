import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model import ExoplanetModel


def test_model_initialization():
    model = ExoplanetModel()
    assert model.model is None
    assert model.preprocessor is None


def test_feature_importance():
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import RobustScaler
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer

    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=8)
    X_df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(10)])

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[("num", numeric_transformer, X_df.columns.tolist())],
        remainder="drop"
    )

    le = LabelEncoder()
    y_encoded = le.fit_transform(["class_0", "class_1", "class_2"] * 34)[:100]

    model = ExoplanetModel()

    params = {
        "n_estimators": 10,
        "max_depth": 3,
        "random_state": 42
    }

    model.fit(X_df, y_encoded, preprocessor, le, X_df.columns.tolist(), params)

    importance = model.get_feature_importance(top_k=5)
    assert len(importance) == 5
    assert all(isinstance(v, float) for v in importance.values())
