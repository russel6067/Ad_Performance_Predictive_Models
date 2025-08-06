import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from scripts.data_preprocessing import load_data, split_features_target
from scripts.evaluation import evaluate_model, plot_actual_vs_predicted, compare_models

# Load and preprocess data
df = load_data()

st.title("Ad Performance Predictive Models")

# Show dataframe
if st.checkbox("Show Raw Data"):
    st.dataframe(df)

# Select target and feature columns
numeric_cols = df.select_dtypes(include='number').columns.tolist()
target_col = st.selectbox("Select target column:", numeric_cols, index=numeric_cols.index('Conversions') if 'Conversions' in numeric_cols else 0)
feature_cols = st.multiselect("Select feature columns:", [col for col in numeric_cols if col != target_col], default=[col for col in numeric_cols if col != target_col])

# Proceed only if features selected
if not feature_cols:
    st.warning("Please select at least one feature column.")
    st.stop()

# Split data
X, y = split_features_target(df[[*feature_cols, target_col]], target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    y_pred, mae, rmse, r2, mse = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MSE': mse}

    st.subheader(f"{name} - Results")
    st.write(f"**MAE**: {mae:.2f}")
    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**R²**: {r2:.4f}")
    st.write(f"**MSE**: {mse:.2f}")

    # Plot actual vs predicted
    fig = plot_actual_vs_predicted(y_test, y_pred, title=f"{name} - Actual vs Predicted")
    st.pyplot(fig)

# Compare all models
st.subheader("📊 Model Comparison")
comparison_df = pd.DataFrame(results).T.sort_values(by="R²", ascending=False)
st.dataframe(comparison_df.style.highlight_max(color='lightgreen', axis=0))

# Best model based on R²
best_model_name = comparison_df['R²'].idxmax()
st.success(f"✅ **Best Model Based on R²**: {best_model_name}")
