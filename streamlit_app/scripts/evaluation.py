from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return y_pred, mae, rmse, r2, mse


def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    return fig


def compare_models(results_dict, metric="R²"):
    import pandas as pd
    df = pd.DataFrame(results_dict).T
    return df.sort_values(by=metric, ascending=False)
