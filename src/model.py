from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import joblib
import shap
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


class ExoplanetModel:
    def __init__(self, model_path: str = None):
        self.model = None
        self.preprocessor = None
        self.label_encoder = None
        self.feature_names = None
        self.explainer = None

        if model_path:
            self.load(model_path)

    def fit(self, X, y, preprocessor, label_encoder, feature_names, params: dict):
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder
        self.feature_names = feature_names

        X_processed = preprocessor.fit_transform(X)

        self.model = lgb.LGBMClassifier(**params, random_state=params.get("random_state", 42))
        self.model.fit(X_processed, y)

        self.explainer = shap.TreeExplainer(self.model)

        return self

    def predict(self, X) -> np.ndarray:
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)

    def predict_proba(self, X) -> np.ndarray:
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)

    def explain(self, X, top_k: int = 5) -> Dict:
        X_processed = self.preprocessor.transform(X)
        shap_values = self.explainer.shap_values(X_processed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        explanations = []
        for i in range(len(X)):
            sample_shap = shap_values[i] if len(shap_values.shape) == 2 else shap_values[i, :, 1]

            importance_idx = np.argsort(np.abs(sample_shap))[-top_k:][::-1]

            top_features = {
                self.feature_names[idx]: float(sample_shap[idx])
                for idx in importance_idx
            }
            explanations.append(top_features)

        return explanations

    def get_feature_importance(self, top_k: int = 10) -> Dict[str, float]:
        importance = self.model.feature_importances_
        sorted_idx = np.argsort(importance)[-top_k:][::-1]

        return {
            self.feature_names[idx]: float(importance[idx])
            for idx in sorted_idx
        }

    def save(self, model_dir: str, timestamp: str):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_dir / f"model_{timestamp}.joblib")
        joblib.dump(self.preprocessor, model_dir / f"preprocessor_{timestamp}.joblib")
        joblib.dump(self.label_encoder, model_dir / f"label_encoder_{timestamp}.joblib")
        joblib.dump(self.feature_names, model_dir / f"features_{timestamp}.joblib")

    def load(self, model_path: str):
        model_path = Path(model_path)
        timestamp = model_path.stem.split("_")[-1]
        model_dir = model_path.parent

        self.model = joblib.load(model_dir / f"model_{timestamp}.joblib")
        self.preprocessor = joblib.load(model_dir / f"preprocessor_{timestamp}.joblib")
        self.label_encoder = joblib.load(model_dir / f"label_encoder_{timestamp}.joblib")
        self.feature_names = joblib.load(model_dir / f"features_{timestamp}.joblib")

        self.explainer = shap.TreeExplainer(self.model)
