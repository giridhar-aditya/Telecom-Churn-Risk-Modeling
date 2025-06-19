import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle

# Load test dataset
df_test = pd.read_csv('dataset.csv')  # replace with your test CSV path
df_test['TotalCharges'] = pd.to_numeric(df_test['TotalCharges'], errors='coerce').fillna(0)
df_test.drop('customerID', axis=1, inplace=True)
df_test['Churn'] = df_test['Churn'].map({'No': 0, 'Yes': 1})

X_test = df_test.drop('Churn', axis=1)
y_test = df_test['Churn']

cat_cols = X_test.select_dtypes(include=['object']).columns.tolist()

# Apply saved target encodings
def apply_target_encoding(series, map_path):
    with open(map_path, 'rb') as f:
        mapping = pickle.load(f)
    prior = np.mean(list(mapping.values()))
    encoded = series.map(mapping)
    encoded.fillna(prior, inplace=True)
    return encoded

for col in cat_cols:
    X_test[col] = apply_target_encoding(X_test[col], f'target_encoding_{col}.pkl')

# Load model
model = xgb.Booster()
model.load_model('xgb_churn_model.json')

dtest = xgb.DMatrix(X_test)

# Predict and evaluate
preds_proba = model.predict(dtest)
preds = (preds_proba > 0.5).astype(int)

print("Test Accuracy:", accuracy_score(y_test, preds))
print("\nTest Classification Report:")
print(classification_report(y_test, preds))
print("Test ROC AUC Score:", roc_auc_score(y_test, preds_proba))
