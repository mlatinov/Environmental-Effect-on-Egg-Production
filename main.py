
#### Libraries ####
import pandas as pd

## Import custom functions ##
from functions.eda_f import eda
from functions.process_data_f import process_data
from functions.tune_models_f import *
from functions.helpf import *

## Load the data ##
egg_data = pd.read_csv("data/Egg_Production.csv")

## Basic EDA ##
plots = eda(data = egg_data)

## Process the Data for ML ##
processing_output = process_data(data=egg_data, Y="Total_egg_production")

## Tune models ##
rf_model = tune_rf(processing_output = processing_output, n_trials=20) # Random Forest model
knn_model = tune_knn(processing_output= processing_output,n_trials=20) # KNN
svm_model = svm_tune(processing_output = processing_output, n_trials= 100) # SVM Model
hist_boost_model = hist_boost_tune(processing_output = processing_output, n_trails=100) # Histboost

## Predict with the  models on the testing data and evaluate them
models_evaluation = pd.DataFrame(
    evaluate_models(
        models = [
            ("rf", rf_model["tuned_pipeline"]),
            ("knn", knn_model["tuned_pipeline"]),
            ("svm", svm_model["tuned_pipeline"]),
            ("histboost", hist_boost_model["tuned_pipeline"])],
        processing_output= processing_output
    )
)
