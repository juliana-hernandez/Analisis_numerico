import numpy as np
from src.application.numerical_method.interfaces.interpolation_method import InterpolationMethod
from src.application.shared.utils.plot_spline import plot_spline_linear


class SplineLinearService(InterpolationMethod):
    def solve(
        self,
        x: list[float],
        y: list[float],
    ) -> dict:
        n = len(x)
        if n < 2:
            return {
                "message_method": "Se necesitan al menos 2 puntos para calcular un spline lineal.",
                "is_successful": False,
                "have_solution": False,
                "tramos": [],
            }

        tramos = []
        equations = []

        for i in range(n - 1):
            # Calcular la pendiente (m)
            m = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
            # Construir el polinomio en formato evaluable
            tramo = f"{m:.4f}*(x - ({x[i]:.4f})) + {y[i]:.4f}"
            tramos.append(tramo)
            equations.append(f"Tramo {i + 1}: {tramo}")

        # Generar la gráfica del spline
        points = list(zip(x, y))
        sorted_points = sorted(points, key=lambda point: point[0])
        plot_spline_linear(sorted_points)

        return {
            "message_method": "Spline lineal calculado con éxito.",
            "is_successful": True,
            "have_solution": True,
            "tramos": tramos,
            "equations": equations,
        }

    def validate_input(
        self, x_input: str, y_input: str
    ) -> str | list[tuple[float, float]]:
        max_points = 8

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
