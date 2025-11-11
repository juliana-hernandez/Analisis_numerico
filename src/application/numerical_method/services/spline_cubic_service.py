import numpy as np
from src.application.numerical_method.interfaces.interpolation_method import InterpolationMethod
from scipy.interpolate import CubicSpline
from src.application.shared.utils.plot_spline import plot_spline_cubic


class SplineCubicService(InterpolationMethod):
    
    def solve(self, x: list[float], y: list[float]) -> dict:
        if len(x) < 3:
            return {
                "message_method": "Se necesitan al menos 3 puntos para calcular un spline cúbico.",
                "is_successful": False,
                "have_solution": False,
            }
        
        # Asegurar que los datos estén ordenados por los valores de x
        sorted_points = sorted(zip(x, y), key=lambda point: point[0])
        x = [point[0] for point in sorted_points]
        y = [point[1] for point in sorted_points]
        
        # Crear el spline cúbico con scipy
        cs = CubicSpline(x, y, bc_type='natural')

        # Obtener los coeficientes del spline (a, b, c, d)
        coefs = cs.c.T  # Coeficientes organizados por tramo
        tramos = []
        for i in range(len(coefs)):
            tramo = (
                f"{coefs[i, 0]:.4f} + {coefs[i, 1]:.4f}*(x - {x[i]:.4f}) "
                f"+ {coefs[i, 2]:.4f}*(x - {x[i]:.4f})^2 + {coefs[i, 3]:.4f}*(x - {x[i]:.4f})^3"
            )
            tramos.append(tramo)

        # Generar la gráfica del spline usando scipy
        plot_spline_cubic("Spline Cúbico", list(zip(x, y)), x, y)

        return {
            "message_method": "Spline cúbico calculado con éxito.",
            "is_successful": True,
            "have_solution": True,
            "tramos": tramos,
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
