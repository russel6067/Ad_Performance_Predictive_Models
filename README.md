# 🎯 Ad Performance Predictive Models

This project compares multiple machine learning models to predict digital ad performance metrics: CTR, CPC, and ROAS.

## 📁 Folder Structure

AD_PERFORMANCE_PREDICTIVE_MODELS/
│
├── data/
│ └── merged_ads_data.csv
│
├── notebooks/
│ └── Predictive models.ipynb
│
├── scripts/
│ ├── data_preprocessing.py
│ ├── evaluation.py
│ ├── model_training.py
│ └── visualization.py
│
├── results/
│ └── model_performance.csv
│
├── streamlit_app/
│ └── app.py
│
├── requirements.txt
└── README.md

yaml
Copy
Edit

## 🚀 How to Run

1. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    venv\Scripts\activate  # for Windows
    ```

2. Install requirements:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run streamlit_app/app.py
    ```

## 📌 Features
- User-selectable target variable: CTR, CPC, or ROAS.
- Trains and compares: Linear Regression, Decision Tree, Random Forest, XGBoost.
- Evaluation metrics: MAE, RMSE, R².
- Visual comparison: Predicted vs Actual scatterplots + bar chart.
- Auto-save results as CSV.