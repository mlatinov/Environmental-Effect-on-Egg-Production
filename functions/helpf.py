from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_models(models, processing_output):
    results = []
    X_train = processing_output["X_train"]
    Y_train = processing_output["Y_train"]
    X_test = processing_output["X_test"]
    Y_test = processing_output["Y_test"]

    # Loop over the models and evaluate each one
    for name, model in models.items():
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # RMSE
        rmse_train = np.sqrt(mean_squared_error(Y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(Y_test, y_test_pred))

        # Return the results from each one separately
        results.append({
            "model": name,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
        })

    return results