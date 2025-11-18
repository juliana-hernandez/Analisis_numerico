import numpy as np
import time
from src.application.numerical_method.interfaces.matrix_method import MatrixMethod
from src.application.shared.utils.plot_matrix_solution import plot_matrix_solution, plot_system_equations


class GaussSeidelService(MatrixMethod):
    def solve(
        self,
        A: list[list[float]],  # Matriz de coeficientes
        b: list[float],  # Vector de términos independientes
        x0: list[float],  # Vector inicial de aproximación
        tolerance: float,  # Tolerancia para el error
        max_iterations: int,  # Número máximo de iteraciones
        precision: int,  # Tipo de precisión (1 para decimales correctos, 0 para cifras significativas)
        **kwargs,
    ) -> dict:

        A = np.array(A)
        b = np.array(b)
        x0 = np.array(x0)

        n = len(b)
        x1 = np.zeros_like(x0)
        current_error = float("inf")
        current_iteration = 0
        table = {}
        show_error_report = kwargs.get("show_error_report", False)
        error_entries = []

        # Inicialización de matrices para el cálculo de T y C
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)

        # Cálculo de la matriz de iteración T para el método Gauss-Seidel
        T = np.linalg.inv(D - L).dot(U)
        spectral_radius = max(abs(np.linalg.eigvals(T)))

        x_prev = x0.copy()
        while current_iteration < max_iterations:
            iter_start = time.perf_counter()

            # Iteración de Gauss-Seidel (actualizamos en x1 usando valores nuevos para índices < i)
            x1 = x_prev.copy()
            for i in range(n):
                sum_others = np.dot(A[i, :i], x1[:i]) + np.dot(A[i, i + 1:], x_prev[i + 1:])
                x1[i] = (b[i] - sum_others) / A[i, i]

            # Diferencia entre iteraciones
            diff = x1 - x_prev

            # Calcular métricas de error
            with np.errstate(divide="ignore", invalid="ignore"):
                arr_abs = np.abs(diff)
                val_abs = float(np.nanmax(arr_abs)) if arr_abs.size > 0 else 0.0

                denom1 = np.abs(x1)
                arr_rel1 = np.where(denom1 != 0, np.abs(diff) / denom1, np.inf)
                val_rel1 = float(np.nanmax(arr_rel1))

                denom2 = np.abs(x_prev)
                arr_rel2 = np.where(denom2 != 0, np.abs(diff) / denom2, np.inf)
                val_rel2 = float(np.nanmax(arr_rel2))

                arr_rel3 = np.abs(diff) * np.abs(x1)
                val_rel3 = float(np.nanmax(arr_rel3))

                arr_rel4 = np.abs(x1)
                val_rel4 = float(np.nanmax(arr_rel4))

            iter_end = time.perf_counter()
            iter_duration = iter_end - iter_start

            # Aplicar precisión según el tipo seleccionado
            if precision == 1:  # Decimales correctos
                decimals = len(str(tolerance).split(".")[1]) if "." in str(tolerance) else 0
                x1_rounded = [round(value, decimals) for value in x1]
                error_rounded = round(float(np.nanmax(np.abs(diff))), decimals)
            elif precision == 0:  # Cifras significativas
                significant_digits = len(str(tolerance).replace("0.", ""))
                x1_rounded = [float(f"{value:.{significant_digits}g}") for value in x1]
                error_rounded = float(f"{float(np.nanmax(np.abs(diff))):.{significant_digits}g}")
            else:
                x1_rounded = x1.tolist()
                error_rounded = float(np.nanmax(np.abs(diff)))

            # Guardamos la información de la iteración actual
            table[current_iteration + 1] = {
                "iteration": current_iteration + 1,
                "X": x1_rounded,
                "Error": error_rounded,
            }

            # Añadir informe de errores plano si se pidió
            if show_error_report:
                error_entries.extend([
                    {"iteration": current_iteration + 1, "type": "Error absoluto", "value": val_abs, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 1", "value": val_rel1, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 2", "value": val_rel2, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 3", "value": val_rel3, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 4", "value": val_rel4, "time_elapsed": iter_duration},
                ])

            # Preparación para la siguiente iteración
            current_iteration += 1
            x_prev = x1.copy()

            if float(np.linalg.norm(diff, ord=np.inf)) <= tolerance:
                break

        # Verificación de éxito o fallo tras las iteraciones
        result = {}
        if current_error <= tolerance:
            result = {
                "message_method": f"Aproximación de la solución con tolerancia = {tolerance} y el radio espectral es de = {spectral_radius}",
                "table": table,
                "is_successful": True,
                "have_solution": True,
                "solution": x1_rounded,
                "spectral_radius": spectral_radius,
            }
        elif current_iteration >= max_iterations:
            result = {
                "message_method": f"El método funcionó correctamente, pero no se encontró una solución en {max_iterations} iteraciones y el radio espectral es de = {spectral_radius}.",
                "table": table,
                "is_successful": True,
                "have_solution": False,
                "solution": x1_rounded,
                "spectral_radius": spectral_radius,
            }
        else:
            result = {
                "message_method": f"El método falló al intentar aproximar una solución",
                "table": table,
                "is_successful": True,
                "have_solution": False,
                "solution": [],
            }

        # Si la matriz es 2x2, generar las gráficas
        if len(A) == 2:
            plot_matrix_solution(table, x1_rounded, spectral_radius)
            plot_system_equations(A.tolist(), b.tolist(), x1_rounded)

        if show_error_report:
            result["error_entries"] = error_entries

        return result

    def validate_input(
        self,
        matrix_a_raw: str,
        vector_b_raw: str,
        initial_guess_raw: str,
        tolerance: float,
        max_iterations: int,
        matrix_size: int,
        **kwargs,
    ) -> str | list:

        # Validación de los parámetros de entrada tolerancia positiva
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            return "La tolerancia debe ser un número positivo"

        # Validación de los parámetros de entrada maximo numero de iteraciones positivo
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            return "El máximo número de iteraciones debe ser un entero positivo."

        # Validación de las entradas numéricas
        try:
            A = [
                [float(num) for num in row.strip().split()]
                for row in matrix_a_raw.split(";")
                if row.strip()
            ]

            b = [float(num) for num in vector_b_raw.strip().split()]
            x0 = [float(num) for num in initial_guess_raw.strip().split()]
        except ValueError:
            return "Todas las entradas deben ser numéricas."

        # Validar que A es cuadrada y coincide con el tamaño seleccionado
        if len(A) != matrix_size or any(len(row) != matrix_size for row in A):
            return f"La matriz A debe ser cuadrada y coincidir con el tamaño seleccionado ({matrix_size}x{matrix_size})."

        # Validar que A es cuadrada y de máximo tamaño 6x6
        if len(A) > 6 or any(len(row) != len(A) for row in A):
            return "La matriz A debe ser cuadrada de hasta 6x6."

        # Validar que b y x0 tengan tamaños compatibles con A
        if len(b) != len(A) or len(x0) != len(A):
            return (
                "El vector b y x0 deben ser compatibles con el tamaño de la matriz A."
            )

        return [A, b, x0]
