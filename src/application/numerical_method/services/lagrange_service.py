import numpy as np
from src.application.numerical_method.interfaces.interpolation_method import (
    InterpolationMethod,
)
from src.application.shared.utils.build_polynomial import build_polynomial


class LagrangeService(InterpolationMethod):
    def solve(
        self,
        x: list[float],
        y: list[float],
    ) -> dict:
        n = len(x)
        coefficients_table = np.zeros((n, n))

        # Construcción de los polinomios de Lagrange
        for i in range(n):
            Li = np.array([1.0])  # Inicializamos Li como un polinomio constante igual a 1
            denominator = 1.0
            for j in range(n):
                if j != i:
                    # Construcción del término (x - x[j]) y acumulación
                    Li = np.convolve(Li, [1, -x[j]])
                    denominator *= (x[i] - x[j])
            coefficients_table[i, :] = y[i] * Li / denominator

        # Suma de los polinomios Lagrange para obtener el polinomio interpolante
        coefficients = np.sum(coefficients_table, axis=0)

        # Asegurar que los coeficientes estén en orden descendente de grado
        coefficients = coefficients[::-1]

        # Construcción del polinomio en forma de cadena
        polynomial = build_polynomial(coefficients)

        return {
            "message_method": "El polinomio interpolante fue encontrado con éxito.",
            "polynomial": polynomial,
            "is_successful": True,
            "have_solution": True,
        }

    def validate_input(
        self, x_input: str, y_input: str
    ) -> str | list[tuple[float, float]]:
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

        # Validamos que los elementos de x sean únicos.
        if len(set(x_values)) != len(x_values):
            return "Error: Los valores de 'x' deben ser únicos."

        # Verificar que el número de puntos no exceda el límite máximo
        if len(x_values) > max_points:
            return f"Error: El número máximo de puntos es {max_points}."

        return [x_values, y_values]