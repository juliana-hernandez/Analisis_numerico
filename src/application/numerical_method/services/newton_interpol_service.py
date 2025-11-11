import numpy as np
import sympy as sp
from src.application.numerical_method.interfaces.interpolation_method import (
    InterpolationMethod,
)

class NewtonInterpolService(InterpolationMethod):
    def solve(self, x: list[float], y: list[float]) -> dict:
        # Verificar que las listas de entrada tengan el mismo tamaño
        if len(x) != len(y):
            return {
                "message_method": "Error: Las listas de 'x' y 'y' deben tener la misma cantidad de elementos.",
                "polynomial": "",
                "is_successful": False,
                "have_solution": False,
            }

        # Número de puntos
        n = len(x)

        # Crear la tabla de diferencias divididas
        divided_diff_table = np.zeros((n, n))
        divided_diff_table[:, 0] = y  # Colocar y en la primera columna

        # Calcular las diferencias divididas
        for j in range(1, n):
            for i in range(n - j):
                divided_diff_table[i, j] = (
                    divided_diff_table[i + 1, j - 1] - divided_diff_table[i, j - 1]
                ) / (x[i + j] - x[i])

        # Obtener los coeficientes de la primera fila de cada columna
        coefficients = divided_diff_table[0, :]

        # Construir el polinomio simbólico
        x_symbol = sp.symbols("x")
        polynomial = coefficients[0]
        term = 1

        for i in range(1, n):
            term *= (x_symbol - x[i - 1])
            polynomial += coefficients[i] * term

        # Simplificar el polinomio
        simplified_polynomial = sp.simplify(polynomial)

        return {
            "message_method": "El polinomio interpolante fue encontrado con éxito.",
            "polynomial": str(simplified_polynomial),
            "is_successful": True,
            "have_solution": True,
        }

    def validate_input(self, x_input: str, y_input: str) -> str | list[tuple[float, float]]:
        max_points = 10

        # Convertir las cadenas de entrada en listas
        x_list = [value.strip() for value in x_input.split(" ") if value.strip()]
        y_list = [value.strip() for value in y_input.split(" ") if value.strip()]

        # Validar que las listas no estén vacías
        if len(x_list) == 0 or len(y_list) == 0:
            return "Error: Las listas de 'x' y 'y' no pueden estar vacías."

        # Validar que ambas listas tengan el mismo tamaño
        if len(x_list) != len(y_list):
            return "Error: Las listas de 'x' y 'y' deben tener la misma cantidad de elementos."

        # Validar que cada elemento de x_list y y_list es numérico
        try:
            x_values = [float(value) for value in x_list]
            y_values = [float(value) for value in y_list]
        except ValueError:
            return "Error: Todos los valores de 'x' y 'y' deben ser numéricos."

        # Validar que los elementos de x sean únicos
        if len(set(x_values)) != len(x_values):
            return "Error: Los valores de 'x' deben ser únicos."

        # Verificar que el número de puntos no exceda el límite máximo
        if len(x_values) > max_points:
            return f"Error: El número máximo de puntos es {max_points}."

        return [x_values, y_values]
