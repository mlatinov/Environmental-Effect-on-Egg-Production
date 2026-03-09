
import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

def tune_rf(processing_output, n_trials):
    """
    Function to tune RF with Optuna
    :param processing_output: The output from the process_data function
    :param n_trials: Times to run the optimization
    :return: Tune Model : model,
            Optuna Study : study ,
            Metainformation : best_parameters, best_score, tune_information_df
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
    # Fir the model to the training data
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




