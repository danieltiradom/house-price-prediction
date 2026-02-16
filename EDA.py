# Exploratory Data Analysis (EDA) | Análisis Exploratorio de Datos
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def run_eda(data):
    # Load and visualize the dataset | Cargar y visualizar el dataset
    print("First rows:")
    print(data.head())

    print("\nDataset info:")
    print(data.info())

    print("\nSummary statistics:")
    print(data.describe())

    # Correlation matrix | Matrix de correlación
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Matrix")
    plt.show()
    plt.close()