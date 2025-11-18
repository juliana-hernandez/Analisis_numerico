import numpy as np
import time
from src.application.numerical_method.interfaces.interpolation_method import (
    InterpolationMethod,
)
from src.application.shared.utils.build_polynomial import build_polynomial


class VandermondeService(InterpolationMethod):
    def solve(
        self,
        x: list[float],
        y: list[float],
        show_error_report: bool = False,
    ) -> dict:
        # Definimos la longitud nxn que va a tener la matriz y la inicializamos en 0.
        n = len(x)
        V = np.zeros((n, n))

        # Llenamos la matriz con los valores de x elevados a la potencia j (matriz de vandermonde).
        for i in range(n):
            for j in range(n):
                V[i, j] = x[i] ** j
        # Resolvemos el sistema de ecuaciones lineales para encontrar los coeficientes del polinomio.
        coefficients = np.linalg.solve(V, y)

        # Construimos el polinomio a partir de los coeficientes obtenidos.
        polynomial = build_polynomial(coefficients)

        result = {
            "message_method": "El polinomio interpolante fué encontrado con exito",
            "polynomial": polynomial,
            "is_successful": True,
            "have_solution": True,
        }

        if show_error_report:
            n = len(x)
            error_entries = []
            for i in range(n):
                start = time.perf_counter()
                x_excl = [x[j] for j in range(n) if j != i]
                y_excl = [y[j] for j in range(n) if j != i]
                m = len(x_excl)
                V = np.zeros((m, m))
                for r in range(m):
                    for c in range(m):
                        V[r, c] = x_excl[r] ** c
                try:
                    coeffs = np.linalg.solve(V, y_excl)
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

        # Validamos que los elementos de x sean unicos.
        if len(set(x_values)) != len(x_values):
            return "Error: Los valores de 'x' deben ser únicos."

        # Verificar que el número de puntos no exceda el límite máximo
        if len(x_values) > max_points:
            return f"Error: El número máximo de puntos es {max_points}."

        return [x_values, y_values]
