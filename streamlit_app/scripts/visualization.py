# streamlit_app/scripts/visualization.py
import matplotlib.pyplot as plt
import streamlit as st

def plot_actual_vs_predicted(y_test, y_pred, model_name):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"{model_name} - Actual vs Predicted")
    st.pyplot(fig)

def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results_df['Model'], results_df['RMSE'], color=['green', 'orange', 'skyblue', 'purple'])
    ax.set_title("Model Comparison (RMSE)")
    ax.set_ylabel("RMSE")
    st.pyplot(fig)
