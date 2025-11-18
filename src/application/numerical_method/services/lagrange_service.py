import numpy as np
import time
from src.application.numerical_method.interfaces.interpolation_method import (
    InterpolationMethod,
)
from src.application.shared.utils.build_polynomial import build_polynomial


class LagrangeService(InterpolationMethod):
    def solve(
        self,
        x: list[float],
        y: list[float],
        show_error_report: bool = False,
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

        result = {
            "message_method": "El polinomio interpolante fue encontrado con éxito.",
            "polynomial": polynomial,
            "is_successful": True,
            "have_solution": True,
        }

        # Generar informe de error Leave-One-Out si se solicitó
        if show_error_report:
            n = len(x)
            error_entries = []
            for i in range(n):
                start = time.perf_counter()
                # construir conjuntos sin el punto i
                x_excl = [x[j] for j in range(n) if j != i]
                y_excl = [y[j] for j in range(n) if j != i]
                m = len(x_excl)
                # construir matriz de vandermonde (potencias crecientes)
                V = np.zeros((m, m))
                for r in range(m):
                    for c in range(m):
                        V[r, c] = x_excl[r] ** c
                try:
                    coeffs = np.linalg.solve(V, y_excl)
                    # evaluar en x[i]
                    xval = x[i]
                    y_pred = sum(coeffs[c] * (xval ** c) for c in range(m))
                    abs_err = abs(y[i] - y_pred)
                    rel_err = abs_err / abs(y[i]) if y[i] != 0 else float("inf")
                except Exception:
                    y_pred = None
                    abs_err = None
                    rel_err = None
                duration = time.perf_counter() - start
                error_entries.append({
                    "iteration": i + 1,
                    "x": x[i],
                    "y": y[i],
                    "predicted": y_pred,
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                    "time_elapsed": duration,
                })
            result["error_entries"] = error_entries

        return result

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