import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from scripts.data_preprocessing import load_data, split_features_target
from scripts.evaluation import evaluate_model, plot_actual_vs_predicted, compare_models

st.title("Ad Performance Predictive Models")

# -----------------------------
# 1️⃣ Dataset Upload or Default
# -----------------------------
st.subheader("Upload Dataset or Use Default")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset loaded successfully!")
else:
    df = load_data()
    st.info("Using default dataset from the project.")

st.write("Dataset Preview:", df.head())

# -----------------------------
# 2️⃣ Target Column Selection
# -----------------------------
st.write("### Please select target column: Click Through Rate (CTR), Cost Per Click (CPC) or Return On Ad Spend (ROAS)")
target_col = st.selectbox(
    "Select target column:",
    options=["CTR", "CPC", "ROAS"]
)

X, y = split_features_target(df, target_col)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3️⃣ Define Models
# -----------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

results = {}

# -----------------------------
# 4️⃣ Train and Evaluate Models with Side-by-Side Charts
# -----------------------------
st.subheader("Actual vs Predicted Comparison")

cols = st.columns(2)  # Create two columns for side-by-side charts
i = 0

for name, model in models.items():
    y_pred, mae, rmse, r2, mse = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2}

    with cols[i % 2]:  # Alternate between columns
        st.write(f"**{name}**")
        st.pyplot(plot_actual_vs_predicted(y_test, y_pred, title=f"{name}"))

    i += 1

# -----------------------------
# 5️⃣ Compare Models
# -----------------------------
st.subheader("Model Comparison")
comparison_df = compare_models(results, metric="R²")
st.dataframe(comparison_df)

best_model_name = comparison_df.index[0]
st.success(f"Best model based on R²: **{best_model_name}**")