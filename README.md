# 🎯 Ad Performance Predictive Models


This project predicts Click-Through Rate (CTR), Cost Per Click (CPC) and Return On Ad Spend (ROAS) for advertising campaigns using multiple machine learning models.
It allows users to compare model performance, visualize Actual vs Predicted results and determine the best-performing model based on R² scores.

Features
Upload your own advertising dataset (.csv)

Predict CTR, CPC, or ROAS

Compare 4 predictive models:

Linear Regression

Random Forest Regressor

Gradient Boosting Regressor

XGBoost Regressor

Evaluate models using:

R² (Coefficient of Determination)

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Side‑by‑side Actual vs Predicted visualizations for easy comparison

Automatically identifies the best model based on R²

Folder Structure
bash
Copy
Edit
Ad_Performance_Predictive_Models/
│
├── data/                        # Dataset folder (optional for sample data)
│   └── merged_ads_data.csv
│
├── streamlit_app/
│   ├── app.py                    # Main Streamlit app
│   └── scripts/
│       ├── data_preprocessing.py # Data loading & cleaning
│       ├── model_training.py     # Model definitions & training
│       └── evaluation.py         # Evaluation metrics & charts
│
├── requirements.txt              # Dependencies for Streamlit Cloud
├── README.md                     # Project documentation
└── .gitignore                    # Ignore venv & unnecessary files
Installation & Local Setup
Clone the repository:

bash
Copy
Edit
git clone https://github.com/russel6067/Ad_Performance_Predictive_Models.git
cd Ad_Performance_Predictive_Models
Create and activate a virtual environment:

bash
Copy
Edit
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app locally:

bash
Copy
Edit
streamlit run streamlit_app/app.py
Deploy on Streamlit Cloud
Push the project to your GitHub repository.

Go to Streamlit Cloud → New App.

Set the repository and choose:

Branch: main

Main file path: streamlit_app/app.py

Click Deploy.

Streamlit will automatically set up the environment and host your app.

Example Output
Model performance comparison table (R², MSE, RMSE)

Side‑by‑side Actual vs Predicted charts

Best model highlighted based on R²

Requirements
Python 3.8+

pandas

numpy

scikit-learn

xgboost

matplotlib

streamlit