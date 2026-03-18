import mlflow
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Générer un jeu de données de régression synthétique
X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y)

# MLflow tracking
with mlflow.start_run():
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mlflow.log_metric("mse", mse)
    mlflow.sklearn.log_model(model, "model")
