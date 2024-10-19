import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
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

# Custom Transformer for Feature Engineering
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # Convert 'tmsp' to datetime and create new features
        X['tmsp'] = pd.to_datetime(X['tmsp'])
        X['hour_of_day'] = X['tmsp'].dt.hour
        X['day_of_week'] = X['tmsp'].dt.dayofweek

        # Apply the transaction fee calculation
        X['transaction_fee'] = X.apply(calculate_transaction_fee, axis=1)

        # Calculate retries-related features
        X = X.sort_values(['country', 'tmsp'])
        X['is_retry'] = (X['amount'].shift(1) == X['amount']) & (X['tmsp'] - X['tmsp'].shift(1) <= pd.Timedelta(minutes=1))
        X['retry_count'] = X.groupby(['amount', 'country']).cumcount()
        X['is_first_attempt'] = (X['retry_count'] == 0).astype(int)
        X['time_since_first_attempt'] = X.groupby(['amount', 'country'])['tmsp'].transform(lambda x: (x - x.min()).dt.total_seconds())

        # One-hot encode categorical variables except for the target column 'PSP'
        X = pd.get_dummies(X, columns=['country', 'card'], drop_first=True)
        return X

# Create a FeatureEngineering instance
feature_engineering = FeatureEngineering()

# Define other transformers for numerical, categorical, and binary features
numerical_features = ['amount', 'transaction_fee', 'retry_count', 'time_since_first_attempt']
categorical_features = ['3D_secured']  # Encoded 'country' and 'card' already handled in feature engineering
binary_features = ['is_retry', 'is_first_attempt', 'hour_of_day', 'day_of_week']

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Fill missing values with median
    ('scaler', StandardScaler())  # Scale numerical features
])

binary_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value=0))  # Replace missing values for binary features
])

# Combine all preprocessing steps into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('bin', binary_transformer, binary_features)
    ]
)

# Create a final pipeline that integrates feature engineering with the preprocessor
final_pipeline = Pipeline(steps=[
    ('feature_engineering', feature_engineering),  # Apply custom feature engineering first
    ('preprocessor', preprocessor)  # Then apply standard preprocessing
])

# Load the data
data = load_data(DATA_PATH)

# Apply the pipeline and fit_transform on the data
X_transformed = final_pipeline.fit_transform(data)

# Save the entire pipeline including feature engineering and preprocessing
joblib.dump(final_pipeline, 'models/preprocessor.pkl')

print("Feature engineering and preprocessing pipeline saved successfully as 'models/preprocessor.pkl'.")
