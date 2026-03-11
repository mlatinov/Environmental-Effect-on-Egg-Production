
#### Libraries ####
import pandas as pd

## Import custom functions ##
from functions.eda_f import eda
from functions.process_data_f import process_data
from functions.tune_models_f import *

## Load the data ##
egg_data = pd.read_csv("data/Egg_Production.csv")

## Basic EDA ##
plots = eda(data = egg_data)

## Process the Data for ML ##
processing_output = process_data(data=egg_data, Y="Total_egg_production")

## Tune models ##
rf_model = tune_rf(processing_output = processing_output, n_trials=10) # Random Forest model
knn_model = tune_knn(processing_output=processing_output,n_trials=20) # KNN
# svm_model =
# gb_model =
