import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, make_scorer, classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Constants
DATA_PATH = "data/raw/transactions.csv"
FEE_STRUCTURE = {
    'Moneycard': {'success': 5, 'fail': 2},
    'Goldcard': {'success': 10, 'fail': 5},
    'UK_Card': {'success': 3, 'fail': 1},
    'Simplecard': {'success': 1, 'fail': 0.5}
}

# Load and preprocess the data
def load_data(path):
    try:
        data = pd.read_csv(path)
        data = data.drop(columns=['Unnamed: 0'])  # Remove irrelevant column
        return data
    except FileNotFoundError:
        print(f"File not found: {path}")
        return None

# Calculate the transaction fee based on PSP and success status
def calculate_transaction_fee(row):
    psp = row['PSP']
    outcome = 'success' if row['success'] == 1 else 'fail'
    return FEE_STRUCTURE[psp][outcome]

# Custom scoring function to balance success rate and transaction fee
def custom_score(y_true, y_pred, X_features, lambda_weight=0.1):
    success_rate = accuracy_score(y_true, y_pred)
    avg_fee = X_features['transaction_fee'].mean()
    return success_rate - (lambda_weight * avg_fee)

# Helper function to calculate and print evaluation metrics
def print_metrics(y_true, y_pred, model_name, label_encoder):
    print(f"\n{model_name} Performance Metrics:")
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"\nPrecision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Support: {support}")
    print(f"\n*******************************\n")

# Main function to execute the data processing and modeling
def main():
    data = load_data(DATA_PATH)
    if data is None:
        return

    # Convert 'tmsp' to datetime and create new features
    data['tmsp'] = pd.to_datetime(data['tmsp'])
    data['hour_of_day'] = data['tmsp'].dt.hour
    data['day_of_week'] = data['tmsp'].dt.dayofweek

    # Apply the transaction fee calculation
    data['transaction_fee'] = data.apply(calculate_transaction_fee, axis=1)

    # Calculate retries-related features
    data = data.sort_values(['country', 'tmsp'])
    data['is_retry'] = (data['amount'].shift(1) == data['amount']) & (data['tmsp'] - data['tmsp'].shift(1) <= pd.Timedelta(minutes=1))
    data['retry_count'] = data.groupby(['amount', 'country']).cumcount()
    data['is_first_attempt'] = (data['retry_count'] == 0).astype(int)
    data['time_since_first_attempt'] = data.groupby(['amount', 'country'])['tmsp'].transform(lambda x: (x - x.min()).dt.total_seconds())

    # One-hot encode categorical variables except for the target column 'PSP'
    data_encoded = pd.get_dummies(data, columns=['country', 'card'], drop_first=True)

    # Encode the target variable 'PSP'
    label_encoder = LabelEncoder()
    data_encoded['PSP_encoded'] = label_encoder.fit_transform(data['PSP'])

    # Save the label encoder
    joblib.dump(label_encoder, 'models/label_encoder.pkl')

    # Define features and target variable
    features = [
        'amount', '3D_secured', 'transaction_fee', 'hour_of_day', 'day_of_week',
        'retry_count', 'is_retry', 'is_first_attempt', 'time_since_first_attempt'
    ] + [col for col in data_encoded.columns if col.startswith('country_') or col.startswith('card_')]
    target = 'PSP_encoded'

    # Split the data into training, validation, and test sets
    X = data_encoded[features]
    y = data_encoded[target]
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Save the scaler
    joblib.dump(scaler, 'models/scaler.pkl')

    # Prepare X_features DataFrame for custom scoring function
    X_features = X_train.copy()
    X_features['PSP'] = y_train

    # Define the custom scorer for cross-validation
    custom_scorer = make_scorer(custom_score, X_features=X_features, lambda_weight=0.1, greater_is_better=True)

    # Train and evaluate Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    y_val_pred_log = log_reg.predict(X_val_scaled)
    y_test_pred_log = log_reg.predict(X_test_scaled)
    print(f"Logistic Regression Validation Accuracy: {accuracy_score(y_val, y_val_pred_log):.4f}")
    print(f"Logistic Regression Test Accuracy: {accuracy_score(y_test, y_test_pred_log):.4f}")
    print_metrics(y_test, y_test_pred_log, "Logistic Regression", label_encoder)
    joblib.dump(log_reg, 'models/baseline_model.pkl')

    # Train and evaluate Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    y_val_pred_rf = rf_model.predict(X_val_scaled)
    y_test_pred_rf = rf_model.predict(X_test_scaled)
    print(f"Random Forest Custom Score: {cross_val_score(rf_model, X_train_scaled, y_train, cv=5, scoring=custom_scorer).mean():.4f}")
    print(f"Random Forest Validation Accuracy: {accuracy_score(y_val, y_val_pred_rf):.4f}")
    print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_test_pred_rf):.4f}")
    print_metrics(y_test, y_test_pred_rf, "Random Forest", label_encoder)
    joblib.dump(rf_model, 'models/accurate_model.pkl')

if __name__ == "__main__":
    main()