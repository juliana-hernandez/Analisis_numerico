import numpy as np
import time
import sympy as sp
from src.application.numerical_method.interfaces.interpolation_method import (
    InterpolationMethod,
)

class NewtonInterpolService(InterpolationMethod):
    def solve(self, x: list[float], y: list[float], x_extra: float | None = None, y_extra: float | None = None, show_error_report: bool = False) -> dict:
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

        result = {
            "message_method": "El polinomio interpolante fue encontrado con éxito.",
            "polynomial": str(simplified_polynomial),
            "is_successful": True,
            "have_solution": True,
        }

        # Si se proporciona un dato adicional (x_{n+1}, y_{n+1}) calcular error de truncamiento
        if x_extra is not None and y_extra is not None:
            try:
                # Evaluar el polinomio en x_extra usando SymPy (más seguro que eval)
                x_symbol = sp.symbols("x")
                y_pred_sym = simplified_polynomial.subs(x_symbol, x_extra)
                # Convertir a float (N) y calcular diferencia
                y_pred = float(sp.N(y_pred_sym))
                trunc_error = abs(float(y_extra) - y_pred)
                result["truncation_error"] = trunc_error
                result["truncation_predicted"] = y_pred
                result["x_extra"] = x_extra
                result["y_extra"] = y_extra
            except Exception as e:
                # No fallamos el método por la gráfica; sólo notificamos el problema
                result["truncation_error"] = None
                result["truncation_predicted"] = None
                result["x_extra"] = x_extra
                result["y_extra"] = y_extra
                result["truncation_message"] = f"No fue posible calcular el error de truncamiento: {e}"

        # Leave-One-Out error report: para cada punto, construir interpolante sin ese punto
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
