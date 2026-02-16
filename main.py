# Main script to run EDA and training | Script principal para ejecutar EDA y entrenamiento
import pandas as pd
from EDA import run_eda
from train import run_train

# Load the dataset | Cargar el dataset
data = pd.read_csv("data/houses.csv")

# Run EDA and training functions | Ejecutar funciones de EDA y entrenamiento
#run_eda(data)
run_train(data)