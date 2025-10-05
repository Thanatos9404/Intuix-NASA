import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support, roc_curve
)
from imblearn.over_sampling import SMOTE
import yaml

from src.data_loader import load_all_data
from src.features import engineer_features, select_features
from src.preprocess import create_preprocessing_pipeline, prepare_data
from src.model import ExoplanetModel


def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


def evaluate_model(model, X, y, label_encoder, config):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    metrics = {}
    metrics["classification_report"] = classification_report(
        y, y_pred, target_names=label_encoder.classes_, output_dict=True
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y, y_pred, average="weighted"
    )
    metrics["precision"] = float(precision)
    metrics["recall"] = float(recall)
    metrics["f1"] = float(f1)

    if len(label_encoder.classes_) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y, y_proba[:, 1]))
    else:
        metrics["roc_auc"] = float(roc_auc_score(y, y_proba, multi_class="ovr"))

    cm = confusion_matrix(y, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def plot_confusion_matrix(cm, classes, timestamp):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    output_dir = Path("models") / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"confusion_matrix_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_proba, label_encoder, timestamp):
    plt.figure(figsize=(8, 6))
    if len(label_encoder.classes_) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    else:
        from sklearn.preprocessing import label_binarize
        y_bin = label_binarize(y_true, classes=range(len(label_encoder.classes_)))
        for i in range(len(label_encoder.classes_)):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
            plt.plot(fpr, tpr, label=f"{label_encoder.classes_[i]} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    output_dir = Path("models") / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f"roc_curve_{timestamp}.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    config = load_config()
    np.random.seed(config["model"]["random_state"])

    print("Loading and merging datasets...")
    df = load_all_data(config)
    print("\nEngineering features...")
    df = engineer_features(df, config)
    print("\nSelecting features...")
    feature_cols = select_features(df, config)
    print(f"Selected {len(feature_cols)} features")
    print("\nPreparing data...")
    X_df, y, label_encoder = prepare_data(df, feature_cols, config)

    # Always keep X as a DataFrame so feature names are preserved
    X_train_val_df, X_test_df, y_train_val, y_test = train_test_split(
        X_df, y, test_size=config["preprocessing"]["test_size"],
        stratify=y, random_state=config["model"]["random_state"]
    )

    print("\nCreating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(feature_cols, config)

    print("\nTraining model with cross-validation...")
    cv = StratifiedKFold(
        n_splits=config["model"]["cv_folds"],
        shuffle=True,
        random_state=config["model"]["random_state"]
    )
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_val_df, y_train_val), 1):
        X_train_fold_df = X_train_val_df.iloc[train_idx]
        X_val_fold_df = X_train_val_df.iloc[val_idx]
        y_train_fold = y_train_val[train_idx]
        y_val_fold = y_train_val[val_idx]
        preprocessor_fold = create_preprocessing_pipeline(feature_cols, config)
        X_train_processed = preprocessor_fold.fit_transform(X_train_fold_df)
        X_val_processed = preprocessor_fold.transform(X_val_fold_df)
        if config["smote"]["enabled"]:
            smote = SMOTE(
                sampling_strategy=config["smote"]["sampling_strategy"],
                k_neighbors=config["smote"]["k_neighbors"],
                random_state=config["model"]["random_state"]
            )
            X_train_processed, y_train_fold_res = smote.fit_resample(X_train_processed, y_train_fold)
        else:
            y_train_fold_res = y_train_fold
        import lightgbm as lgb
        model_fold = lgb.LGBMClassifier(**config["model"]["lgb_params"])
        model_fold.fit(X_train_processed, y_train_fold_res)
        val_pred = model_fold.predict(X_val_processed)
        from sklearn.metrics import f1_score
        fold_score = f1_score(y_val_fold, val_pred, average="weighted")
        cv_scores.append(fold_score)
        print(f"Fold {fold} F1: {fold_score:.4f}")

    print(f"\nCV F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
    print("\nTraining final model on full train+val set...")
    final_preprocessor = create_preprocessing_pipeline(feature_cols, config)
    X_train_val_processed = final_preprocessor.fit_transform(X_train_val_df)
    # Apply SMOTE on both feature matrix AND target and use results for model fitting
    if config["smote"]["enabled"]:
        smote = SMOTE(
            sampling_strategy=config["smote"]["sampling_strategy"],
            k_neighbors=config["smote"]["k_neighbors"],
            random_state=config["model"]["random_state"]
        )
        X_train_val_processed, y_train_val_res = smote.fit_resample(X_train_val_processed, y_train_val)
    else:
        y_train_val_res = y_train_val
    # For feature names with pipeline transform, use column indices
    final_model = ExoplanetModel()
    final_model.fit(
        X_train_val_processed, y_train_val_res, final_preprocessor,
        label_encoder, feature_cols, config["model"]["lgb_params"]
    )
    print("\nEvaluating on test set...")
    # Preprocess test set for evaluation
    X_test_processed = final_preprocessor.transform(X_test_df)
    test_metrics = evaluate_model(final_model, X_test_processed, y_test, label_encoder, config)
    print("\nTest Metrics:")
    print(f"Precision: {test_metrics['precision']:.4f}")
    print(f"Recall: {test_metrics['recall']:.4f}")
    print(f"F1: {test_metrics['f1']:.4f}")
    print(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\nSaving model and artifacts...")
    final_model.save("models", timestamp)
    with open(f"models/metrics_{timestamp}.json", "w") as f:
        json.dump(test_metrics, f, indent=2)
    plot_confusion_matrix(
        np.array(test_metrics["confusion_matrix"]),
        label_encoder.classes_,
        timestamp
    )
    y_test_proba = final_model.predict_proba(X_test_processed)
    plot_roc_curve(y_test, y_test_proba, label_encoder, timestamp)
    feature_importance = final_model.get_feature_importance(top_k=20)
    print("\nTop 20 Feature Importances:")
    for feat, imp in feature_importance.items():
        print(f"{feat}: {imp:.4f}")
    with open(f"models/feature_importance_{timestamp}.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    print(f"\nTraining complete! Model saved with timestamp: {timestamp}")


if __name__ == "__main__":
    main()
