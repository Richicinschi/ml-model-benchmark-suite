# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-04-02

### Overview
Final release of the ML Model Benchmark Suite - a comprehensive framework for training, evaluating, and comparing machine learning models.

### Features

#### Foundation (Commits 1-5)
- Project structure and configuration system
- Base abstractions for experiments and models
- Config-driven experiment runner core
- Logging and utility modules

#### Data Pipeline (Commits 6-11)
- CSV dataset loader with validation
- Sklearn built-in datasets integration
- OpenML dataset loader and caching
- Preprocessing pipeline: scaling, encoding, imputation

#### Classification Models (Commits 12-21)
- Logistic regression and KNN classifiers
- Random forest and SVM classifiers
- XGBoost classifier integration with GPU support
- Stratified k-fold cross-validation
- GridSearchCV and RandomizedSearchCV hyperparameter tuning
- Classification metrics: accuracy, F1, AUC-ROC

#### Regression Models (Commits 22-30)
- Linear, ridge, lasso, ElasticNet regression
- Gradient boosting regressor
- XGBoost regressor with GPU support
- K-fold cross-validation for regression

#### Evaluation & Metrics (Commits 31-41)
- Comprehensive metrics dashboard
- Confusion matrix and calibration curves
- Feature importance comparison
- Overfitting detection (train vs val curves)

#### Explainability & Tracking (Commits 42-50)
- SHAP value analysis for model explainability
- SQLite experiment tracking
- Model comparison plots
- Run history and query capabilities

#### Polish & Release (Commits 51-59)
- Tutorial notebook for user guide
- GitHub Actions CI/CD workflow
- Flake8 and Black linting configuration
- MyPy type-checking configuration
- In-memory dataset caching with lru_cache
- Signal handling for graceful shutdown
- XGBoost GPU configuration fix
- Full audit and cleanup

### Technical Highlights

#### GPU Acceleration
- XGBoost with CUDA support (`device='cuda'`)
- LightGBM with OpenCL support
- Automatic GPU detection and fallback to CPU

#### Signal Handling
- Graceful shutdown on SIGTERM/SIGINT
- Resource cleanup on exit
- Prevention of orphaned processes

#### Testing
- 102 pytest tests covering all modules
- Integration tests for end-to-end pipelines
- Model-specific tests for classification and regression

### Configuration
Example configs provided for:
- Airline delay prediction (2015-2016 datasets)
- Bank marketing dataset
- Hyperparameter tuning examples

### CLI Commands
```bash
python main.py --config config/example.yaml          # Run experiment
python main.py --list-models                         # List models
python main.py --history                             # View history
python main.py --query-model xgboost                 # Query by model
python main.py --export-json exports/run.json        # Export results
```

### Dependencies
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.3.0
- xgboost >= 1.7.0
- lightgbm >= 4.0.0
- shap >= 0.42.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

### Author
DigitalNomad

---
*Back to roasting beans ☕*
