import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

X = df.drop('Churn', axis=1)
y = df['Churn']

cat_cols = X.select_dtypes(include=['object']).columns.tolist()

# Target encoding function with saving
def target_encode(trn_series, tst_series, target, min_samples_leaf=1, smoothing=1, save_path=None):
    temp = pd.concat([trn_series, target], axis=1)
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    smoothing_val = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    prior = target.mean()
    averages[target.name] = prior * (1 - smoothing_val) + averages["mean"] * smoothing_val

    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(averages[target.name].to_dict(), f)

    ft_trn_series = trn_series.map(averages[target.name])
    ft_tst_series = tst_series.map(averages[target.name])
    ft_tst_series.fillna(prior, inplace=True)
    return ft_trn_series, ft_tst_series

# Split dataset 80-20 stratified
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply target encoding and save maps
for col in cat_cols:
    X_train[col], X_test[col] = target_encode(
        X_train[col], X_test[col], y_train, min_samples_leaf=100, smoothing=10,
        save_path=f'target_encoding_{col}.pkl'
    )

# Prepare DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Best hyperparameters found earlier
best_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'booster': 'gbtree',
    'lambda': 0.004069557221941554,
    'alpha': 0.22741153731083966,
    'colsample_bytree': 0.7822418874534431,
    'subsample': 0.38338588656746564,
    'learning_rate': 0.020978806669532626,
    'max_depth': 3,
    'min_child_weight': 3,
    'random_state': 42,
    'verbosity': 1
}

# Train model
model = xgb.train(best_params, dtrain, num_boost_round=1000)

# Predict on test to check performance
preds_proba = model.predict(dtest)
preds = (preds_proba > 0.5).astype(int)

print("Test Accuracy:", accuracy_score(y_test, preds))
print("\nTest Classification Report:")
print(classification_report(y_test, preds))
print("Test ROC AUC Score:", roc_auc_score(y_test, preds_proba))

# Save model
model.save_model('xgb_churn_model.json')
print("\nModel saved as 'xgb_churn_model.json'")
