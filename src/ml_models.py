import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train_models():
    df = pd.read_csv("data/simpy_dataset.csv")

    X = df.drop("avg_wait_time", axis=1)
    y = df["avg_wait_time"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "KNN": KNeighborsRegressor(),
        "SVR": SVR(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results.append([
            name,
            mean_squared_error(y_test, preds),
            mean_absolute_error(y_test, preds),
            r2_score(y_test, preds)
        ])

    results_df = pd.DataFrame(results, columns=["Model", "MSE", "MAE", "R2"])
    results_df.to_csv("results/model_comparison.csv", index=False)
    return results_df

if __name__ == "__main__":
    train_models()
