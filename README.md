# Delivery Delay Prediction
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

## Problem Statement
Predict whether a delivery will miss its promised SLA before dispatch, enabling proactive logistics decisions.

## Dataset
Synthetic dataset generated to simulate real-world logistics behavior including:
- Distance
- Traffic conditions
- Weather
- Courier partner
- Dispatch delays

## Approach
- Binary classification problem
- Logistic Regression used as baseline
- Random Forest used for final model
- Feature engineering focused on operational factors
- Model trained and evaluated using F1-score and ROC-AUC

## Key Insights
- Distance and traffic are the strongest predictors of delay
- Weather conditions significantly increase delay probability
- Random Forest improved recall for delayed deliveries

## Model Artifact
Trained model saved as: model/delivery_delay_model.pkl

## How to Run
1. Open Jupyter Notebook
2. Navigate to `notebooks/03_model_training.ipynb`
3. Run all cells top to bottom

## Future Improvements
- Real-time weather integration
- Model explainability using SHAP
- Web application for prediction
