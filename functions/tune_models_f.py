import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVR


def tune_rf(processing_output, n_trials):
    """
    Tunes a Random Forest Regressor using Optuna hyperparameter optimization.

    This function performs hyperparameter tuning for a Random Forest Regressor model using the Optuna library. It uses cross-validation to evaluate different hyperparameter combinations and selects the best one.

    Parameters:
    - processing_output (dict): A dictionary containing the preprocessing pipeline and training data. Expected keys include 'processing' (the preprocessing step), 'X_train_data', and 'Y_train_data'.
    - n_trials (int): The number of optimization trials to run with Optuna.

    Returns:
    dict: A dictionary containing the tuned model and tuning metadata:
      - 'model': The trained Random Forest Regressor with the best hyperparameters.
      - 'study': The Optuna study object.
      - 'best_parameters': The best hyperparameters found.
      - 'best_score': The best cross-validation score (negative RMSE).
      - 'tune_information_df': A DataFrame with information about all trials.
    """
    # Define objective function
    def objective(trial):
        # Define Pipeline
        pipeline = Pipeline([
            ("process", processing_output["processing"]),
            ("model", RandomForestRegressor(
                # Hyperparameters
                n_estimators=trial.suggest_int("n_estimators", 100, 1000),  # The number of trees in the forest
                max_depth=trial.suggest_int("max_depth", 5, 20),  # The maximum depth of the tree
                min_samples_split=trial.suggest_int("min_samples_split", 2, 40),  # The minimum number of samples
                random_state=42
            ))
        ])
        # Cross Validation
        score = cross_val_score(
            estimator=pipeline,
            X=processing_output["X_train_data"],
            y=processing_output["Y_train_data"],
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        # Return the score from the cross validation
        return -score.mean()

    # Create Study
    study = optuna.create_study(direction="minimize")
    # Run Optimization
    study.optimize(objective,n_trials=n_trials)
    # Get Best Parameters
    best_params = study.best_params
    # Train Final Model
    best_rf = RandomForestRegressor(
        **best_params,
        random_state=42
    )
    # Fit the model to the training data
    fit_rf = best_rf.fit(
        X = processing_output["X_train_data"],
        y = processing_output["Y_train_data"]
    )
    # Store the information about the tuning and the model in a dict
    results = {
        "model" : fit_rf,
        "study" : study,
        "best_parameters" : study.best_params,
        "best_score" : study.best_value,
        "tune_information_df" : study.trials_dataframe()
    }
    # Return the model object with the best parameters
    return results

def tune_knn(processing_output, n_trials):
    """
    Tunes a K-Nearest Neighbors Regressor using Optuna hyperparameter optimization.

    This function performs hyperparameter tuning for a K-Nearest Neighbors Regressor model using the Optuna library. It uses cross-validation to evaluate different hyperparameter combinations and selects the best one.

    Parameters:
    - processing_output (dict): A dictionary containing the preprocessing pipeline and training data. Expected keys include 'processing' (the preprocessing step), 'X_train_data', and 'Y_train_data'.
    - n_trials (int): The number of optimization trials to run with Optuna.

    Returns:
    dict: A dictionary containing the tuned model and tuning metadata:
      - 'model': The trained K-Nearest Neighbors Regressor with the best hyperparameters.
      - 'study': The Optuna study object.
      - 'best_parameters': The best hyperparameters found.
      - 'best_score': The best cross-validation score (negative RMSE).
      - 'tune_information_df': A DataFrame with information about all trials.
    """

    # Define objective function
    def objective(trial):
        # Define Pipeline
      pipeline = Pipeline([
          ("preprocess", processing_output["processing"]),
          ("model", KNeighborsRegressor(
              # Hyperparameters
              n_neighbors=trial.suggest_int("n_neighbors", 2, 20) # Number of neighbors to use
          ))
      ])
      # Cross Validation
      score = cross_val_score(
          estimator=pipeline,
          X=processing_output["X_train_data"],
          y=processing_output["Y_train_data"],
          cv=5,
          scoring="neg_root_mean_squared_error",
          n_jobs=-1
      )
      # Return the score from the cross validation
      return -score.mean()

    # Create a Study
    study = optuna.create_study(direction="minimize")
    #  Run optimization
    study.optimize(objective,n_trials=n_trials)
    # Get the best parameter
    best_parameters = study.best_params
    # Train the final model with the best parameters
    best_knn = KNeighborsRegressor(
        **best_parameters
    )
    # Fit the final models on the train data
    best_knn.fit(
        X=processing_output["X_train_data"],
        y=processing_output["Y_train_data"]
    )
    # Store and return the tuned model and metainformation about the tuning performance
    results = {
        "model" : best_knn,
        "study" : study,
        "best_parameters" : study.best_params,
        "best_score" : study.best_value,
        "tune_information_df" : study.trials_dataframe()
    }
    return results

def svm_tune(processing_output, n_trials):
    """
       Tunes a Support Vector Machine Regressor using Optuna hyperparameter optimization.

       This function performs hyperparameter tuning for a Linear Support Vector Regressor model using the Optuna library. It uses cross-validation to evaluate different hyperparameter combinations and selects the best one.

       Parameters:
       - processing_output (dict): A dictionary containing the preprocessing pipeline and training data. Expected keys include 'processing' (the preprocessing step), 'X_train_data', and 'Y_train_data'.
       - n_trials (int): The number of optimization trials to run with Optuna.

       Returns:
       dict: A dictionary containing the tuned model and tuning metadata:
         - 'model': The trained Linear Support Vector Regressor with the best hyperparameters.
         - 'study': The Optuna study object.
         - 'best_parameters': The best hyperparameters found.
         - 'best_score': The best cross-validation score (negative RMSE).
         - 'tune_information_df': A DataFrame with information about all trials.
       """
    # Define objective function
    def objective(trial):
        # Define a pipeline
        pipeline = Pipeline([
            ("preprocess", processing_output["processing"]),
            ("model", LinearSVR(
                # Hyperparameters
                C=trial.suggest_float("C", 0.1, 100),
                epsilon=trial.suggest_float("epsilon", 0.001, 1)
            ))
        ])
        # Cross validation
        score = cross_val_score(
            estimator=pipeline,
            X=processing_output["X_train_data"],
            y=processing_output["Y_train_data"],
            cv=5,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1
        )
        return -score.mean()
    # Create a study with optuna
    study = optuna.create_study(direction="minimize")
    # Run the optimization
    study.optimize(objective,n_trials=n_trials)
    # Get the best parameters
    best_parameters = study.best_params
    # Train the tuned model with the best parameters
    tuned_svm = LinearSVR(
        **best_parameters,
        random_state=42
    )
    # Fit the model to the training data
    tuned_svm.fit(
        X= processing_output["X_train_data"],
        y=processing_output["Y_train_data"]
    )
    # Store the tuning results and tuned model in dict\
    results = {
        "model" : tuned_svm,
        "study" : study,
        "best_parameters" : study.best_params,
        "best_score" : study.best_value,
        "tune_information_df" : study.trials_dataframe()
    }
    return results

def xgboost_tune(processing_output,n_trails):
    # Define objective function
    def objective(trial):
        # Define a pipeline
        pipeline = Pipeline([
            "processing", processing_output["processing"],
            "model",
        ])
    pass






