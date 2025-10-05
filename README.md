# Exoplanet Detection ML System

Production-ready machine learning system for automated exoplanet identification using NASA's Kepler, K2, and TESS datasets. Features FastAPI inference API, Streamlit web UI, SHAP explainability, and complete CI/CD pipeline.

## Features

- Multi-mission dataset integration (KOI, TOI, K2)
- LightGBM classifier with hyperparameter tuning
- SMOTE-based class imbalance handling
- SHAP-based model explainability
- FastAPI REST API for inference
- Interactive Streamlit web interface
- Automated model retraining capability
- Docker containerization
- GitHub Actions CI/CD

## Quick Start

### Local Setup
```
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Download Data
```
python scripts/download_data.py
```

### Train Model
```
python src/train.py
```


### Run API Server
```
uvicorn src.api:app --host 0.0.0.0 --port 8000
```


### Run Streamlit UI
```
streamlit run src/ui/streamlit_app.py
```

## Docker Deployment

### Build Images
```
docker build -t exoplanet-api -f Dockerfile .
docker build -t exoplanet-ui -f Dockerfile.streamlit .
```

### Run with Docker Compose
```
docker-compose up
```


>Access UI at http://localhost:8501 and API at http://localhost:8000

## Project Structure

exoplanet-ml/
```
├── data/
│ ├── raw/ # Downloaded NASA datasets
│ ├── processed/ # Cleaned and merged data
│ └── sample_input.csv # Example input
├── src/
│ ├── data_loader.py # Data ingestion
│ ├── preprocess.py # Preprocessing pipeline
│ ├── features.py # Feature engineering
│ ├── train.py # Model training
│ ├── model.py # Model wrapper
│ ├── api.py # FastAPI endpoints
│ └── ui/
│ └── streamlit_app.py # Web UI
├── models/ # Saved models and metadata
├── tests/ # Unit tests
├── scripts/
│ └── download_data.py # Data download script
├── config.yaml # Configuration
└── architecture.md # System architecture
```


## API Endpoints

- `POST /predict` - Single record prediction with SHAP explanation
- `POST /predict_batch` - Batch CSV prediction
- `GET /model/metrics` - Current model performance metrics
- `GET /health` - Health check

## Configuration

Edit `config.yaml` to adjust:
- Model hyperparameters
- Cross-validation settings
- Feature engineering options
- SMOTE parameters
- Random seed

## Testing
``````
pytest tests/ -v --cov=src
``````

## Retraining
``````
Via UI: Upload labeled CSV and click "Retrain Model" (max 50k rows)
Via CLI: Place new data in `data/raw/` and run `python src/train.py`
``````
## Model Artifacts

Trained models saved to `models/` with:
- `model_YYYYMMDD_HHMMSS.joblib` - Trained model
- `preprocessor_YYYYMMDD_HHMMSS.joblib` - Preprocessing pipeline
- `metrics_YYYYMMDD_HHMMSS.json` - Performance metrics

## License

>MIT

## Demo
[Exoplanet-Detection-ML-System-Yashvardhan-Thanvi-Intuix.pdf](https://github.com/user-attachments/files/22710996/Exoplanet-Detection-ML-System-Yashvardhan-Thanvi-Intuix.pdf)
