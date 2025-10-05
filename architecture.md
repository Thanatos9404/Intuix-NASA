# System Architecture

## Overview

The exoplanet detection system follows a modular architecture with clear separation between data processing, model training, inference, and user interface layers.

## Components

### Data Layer

**scripts/download_data.py**
- Downloads KOI, TOI, K2 datasets from NASA Exoplanet Archive
- Implements retry logic and error handling
- Stores raw CSVs in `data/raw/`

**src/data_loader.py**
- Loads and validates raw datasets
- Harmonizes column names across missions
- Implements label mapping from mission-specific dispositions

### Processing Layer

**src/preprocess.py**
- Handles missing values (median/mode imputation)
- Removes sparse columns and identifiers
- Scales numeric features (RobustScaler)
- Encodes categorical variables
- Implements sklearn Pipeline for reproducibility

**src/features.py**
- Derives engineered features:
  - Transit depth proxies
  - Period/duration ratios
  - Log transforms of skewed distributions
  - Signal-to-noise estimates
- All transformations are serializable

### Model Layer

**src/train.py**
- Orchestrates training pipeline
- Implements StratifiedKFold cross-validation
- Applies SMOTE only on training folds (no leakage)
- Hyperparameter search (optional)
- Saves models with metadata and timestamps

**src/model.py**
- Model wrapper class with prediction methods
- SHAP explainer integration
- Feature importance computation
- Handles model loading and versioning

### API Layer

**src/api.py**
- FastAPI application with three endpoints:
  - `/predict` - Single sample with SHAP values
  - `/predict_batch` - CSV upload for batch inference
  - `/model/metrics` - Current model statistics
- Automatic request validation via Pydantic
- Error handling and logging

### UI Layer

**src/ui/streamlit_app.py**
- Multi-page Streamlit interface
- Manual entry form with input validation
- Batch upload and download
- Model metrics visualization
- Retraining interface with progress tracking
- SHAP waterfall plots for predictions

## Data Flow

1. **Training Flow**
```
Raw CSVs → data_loader → Label harmonization →
Feature engineering → Train/val/test split →
SMOTE (train only) → LightGBM training →
Cross-validation → Model evaluation →
Save artifacts (model + preprocessor + metrics)
```


2. **Inference Flow**
```
User input (UI/API) → Input validation →
Preprocessor transform → Model predict →
SHAP explanation → Return prediction + probabilities + features
```


3. **Retraining Flow**
```
User uploads labeled CSV → Validation (size/schema) →
Confirmation prompt → Full training pipeline →
Updated model artifacts → Metrics display
```

## Design Decisions

**LightGBM over XGBoost**: Faster training, lower memory, native categorical support

**SMOTE in CV folds**: Prevents data leakage by oversampling only training data

**RobustScaler**: Handles outliers better than StandardScaler for astronomical data

**Joblib serialization**: Efficient for sklearn pipelines, faster than pickle

**FastAPI + Streamlit**: FastAPI for production API, Streamlit for rapid UI prototyping

**Unified labels**: Maps mission-specific dispositions to three classes for consistency

## Scalability Considerations

- Streaming data processing for large CSVs (chunked reading)
- Model versioning with timestamps
- Horizontal scaling via Docker Compose replicas
- Rate limiting in API for production deployment
- Async FastAPI endpoints for concurrent requests

## Security

- Input validation on all endpoints
- CSV size limits (50k rows for retraining)
- No SQL injection risk (CSV-only input)
- Rate limiting recommended for production
- Authentication layer can be added via API keys