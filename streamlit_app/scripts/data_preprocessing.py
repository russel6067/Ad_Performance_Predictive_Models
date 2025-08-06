import os
import pandas as pd

def load_data():
    # Adjusted path to point correctly to the data folder
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'merged_ads_data.csv'))

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")
    
    df = pd.read_csv(data_path)

    # Drop columns that are not useful for modeling (e.g., date/time, IDs)
    df.drop(columns=['date'], inplace=True, errors='ignore')

    # Remove any rows with missing values
    df.dropna(inplace=True)

    # Convert categorical variables to dummy/indicator variables
    df = pd.get_dummies(df, drop_first=True)

    return df

def split_features_target(df, target_column):
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y
