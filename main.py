
#### Libraries ####
import pandas as pd

## Import custom functions ##
from functions.eda_f import eda
from functions.process_data_f import process_data

## Load the data
egg_data = pd.read_csv("data/Egg_Production.csv")

## Basic EDA
plots = eda(data = egg_data)

## Process the Data for ML ##
processing_output = process_data(data=egg_data, Y="Total_egg_production")

