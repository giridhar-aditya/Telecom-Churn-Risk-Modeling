# 📊 Telecom Customer Churn Prediction

A predictive analytics solution leveraging **XGBoost** to accurately identify telecom customers at risk of churn, enabling targeted retention interventions.

## 📂 Dataset

- **Source**: IBM Telco Customer Churn Dataset
- **Rows**: ~7,000 customers
- **Features**: Demographics, account info, usage patterns
- **Target**: `Churn` (Yes/No)

## ⚙️ Model Details

- **Algorithm**: XGBoost Classifier
- **Feature Engineering**: Target Encoding for categorical variables
- **Hyperparameter Tuning**: Optuna optimization
- **Imbalance Handling**: SMOTE for balanced training

## ✅ Performance Metrics

- **Accuracy**: ~76%
- **ROC AUC Score**: **0.84**
- **Classification Report**:
  - Precision (Churn): 0.54
  - Recall (Churn): 0.77
  - F1-Score (Churn): 0.63

> 🔎 **Note**: The primary metric is ROC AUC to balance between precision and recall for churn detection.

## 🗂️ Files

- `dataset.csv` — Cleaned dataset
- `train.py` — Training script with Optuna hyperparameter tuning
- `evaluate.py` — Evaluation script for test datasets
- `xgb_churn_model.json` — Saved XGBoost model
- `target_encoding_<feature>.pkl` — Saved target encoding mappings for reproducibility

## 🚀 Usage

1️⃣ **Train the Model**  
```bash
python train.py
```
2️⃣ Evaluate on Test Data
```bash
python evaluate.py
```
