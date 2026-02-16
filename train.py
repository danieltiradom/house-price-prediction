# Train the model | Entrenar el modelo
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def run_train(data):
    # Select predictor variables | Seleccionar variables predictoras

    features = [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated"
    ]

    x = data[features]
    y = data["price"]

    # Split the dataset into training and testing sets | Dividir el dataset en conjuntos de entrenamiento y prueba
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=49)

    # Create the model | Crear el modelo
    model = LinearRegression()

    # Train the model | Entrenar el modelo
    model.fit(x_train, y_train)

    print("Model trained successfully")
    
    # Make predictions | Hacer predicciones
    y_pred = model.predict(x_test)

    # Evaluate the model | Evaluar el modelo
    mae = mean_absolute_error(y_test, y_pred) # Error absoluto medio, esto significa que en promedio, las predicciones del modelo se desvían del valor real por esta cantidad.
    mse = mean_squared_error(y_test, y_pred) # Error cuadrático medio, esto penaliza más los errores grandes, ya que los errores se elevan al cuadrado. Un valor más bajo indica un mejor rendimiento del modelo.
    r2 = r2_score(y_test, y_pred) # R2 Score, esto indica qué tan bien el modelo explica la variabilidad de los datos. Un valor de 1 indica un modelo perfecto, mientras que un valor de 0 indica que el modelo no explica nada de la variabilidad.

    print("\n---------- Model Evaluation ---------")
    print(f"Mean Absolute Error: {mae}")
    print(f"Mean Squared Error: {mse}")
    print(f"R2 Score: {r2}")

    print("\nModel Interpretation:")
    print("The model explains approximately 51% of the variance in housing prices.")
    print("On average, predictions are off by around $167,000.")

    