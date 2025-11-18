import numpy as np
import time
from src.application.numerical_method.interfaces.matrix_method import MatrixMethod
from src.application.shared.utils.plot_matrix_solution import plot_matrix_solution, plot_system_equations


class SORService(MatrixMethod):
    def solve(
        self,
        A: list[list[float]],  # Matriz de coeficientes
        b: list[float],  # Vector de términos independientes
        x0: list[float],  # Vector inicial de aproximación
        tolerance: float,  # Tolerancia para el error
        max_iterations: int,  # Número máximo de iteraciones
        relaxation_factor: float,  # Factor de relajación (w)
        precision_type: int,  # Tipo de precisión (1 para decimales, 0 para cifras significativas)
        **kwargs,
    ) -> dict:

        A = np.array(A)
        b = np.array(b)
        x0 = np.array(x0)

        n = len(b)
        x = x0.copy()
        table = {}
        show_error_report = kwargs.get("show_error_report", False)
        error_entries = []

        # Inicialización de matrices para el cálculo de T y C
        D = np.diag(np.diag(A))
        L = -np.tril(A, -1)
        U = -np.triu(A, 1)

        # Cálculo de la matriz de iteración T para el método SOR
        T = np.linalg.inv(D - relaxation_factor * L).dot((1 - relaxation_factor) * D + relaxation_factor * U)
        spectral_radius = max(abs(np.linalg.eigvals(T)))

        current_error = tolerance + 1
        current_iteration = 0

        # Iteración SOR
        while current_error > tolerance and current_iteration < max_iterations:
            iter_start = time.perf_counter()
            x_new = x.copy()
            for i in range(n):
                sum_others = np.dot(A[i, :i], x_new[:i]) + np.dot(A[i, i + 1:], x[i + 1:])
                x_new[i] = (1 - relaxation_factor) * x[i] + (relaxation_factor / A[i, i]) * (b[i] - sum_others)

            # Calcular el error como norma infinito de la diferencia
            diff = x_new - x
            current_error = float(np.linalg.norm(diff, ord=np.inf))

            iter_end = time.perf_counter()
            iter_duration = iter_end - iter_start

            # Aplicar precisión al vector de soluciones y al error
            if precision_type == 1:  # Decimales correctos
                decimals = int(-np.floor(np.log10(tolerance))) if tolerance != 0 else 0
                x_new_rounded = np.round(x_new, decimals).tolist()
                error_rounded = round(current_error, decimals)
            elif precision_type == 0:  # Cifras significativas
                factor = 10 ** int(np.ceil(np.log10(abs(1 / tolerance))))
                x_new_rounded = (np.round(x_new * factor) / factor).tolist()
                error_rounded = round(current_error * factor) / factor
            else:
                x_new_rounded = x_new.tolist()
                error_rounded = current_error

            # Guardar información en la tabla para la iteración actual
            table[current_iteration + 1] = {
                "iteration": current_iteration + 1,
                "X": x_new_rounded,
                "Error": error_rounded,
            }

            # Calcular métricas de error solicitadas y agregarlas al informe si se pidió
            if show_error_report:
                with np.errstate(divide="ignore", invalid="ignore"):
                    arr_abs = np.abs(diff)
                    val_abs = float(np.nanmax(arr_abs)) if arr_abs.size > 0 else 0.0

                    denom1 = np.abs(x_new)
                    arr_rel1 = np.where(denom1 != 0, np.abs(diff) / denom1, np.inf)
                    val_rel1 = float(np.nanmax(arr_rel1))

                    denom2 = np.abs(x)
                    arr_rel2 = np.where(denom2 != 0, np.abs(diff) / denom2, np.inf)
                    val_rel2 = float(np.nanmax(arr_rel2))

                    arr_rel3 = np.abs(diff) * np.abs(x_new)
                    val_rel3 = float(np.nanmax(arr_rel3))

                    arr_rel4 = np.abs(x_new)
                    val_rel4 = float(np.nanmax(arr_rel4))

                error_entries.extend([
                    {"iteration": current_iteration + 1, "type": "Error absoluto", "value": val_abs, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 1", "value": val_rel1, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 2", "value": val_rel2, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 3", "value": val_rel3, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 4", "value": val_rel4, "time_elapsed": iter_duration},
                ])

            # Preparar para la siguiente iteración
            x = x_new
            current_iteration += 1

        # Verificación de éxito o fallo tras las iteraciones
        result = {}
        if current_error <= tolerance:
            result = {
                "message_method": f"Aproximación de la solución con tolerancia = {tolerance} y el radio espectral es de = {spectral_radius}",
                "table": table,
                "is_successful": True,
                "have_solution": True,
                "solution": x.tolist(),
                "spectral_radius": spectral_radius,
            }
        elif current_iteration >= max_iterations:
            result = {
                "message_method": f"El método funcionó correctamente, pero no se encontró una solución en {max_iterations} iteraciones y el radio espectral es de = {spectral_radius}.",
                "table": table,
                "is_successful": True,
                "have_solution": False,
                "solution": x.tolist(),
                "spectral_radius": spectral_radius,
            }
        else:
            result = {
                "message_method": f"El método falló al intentar aproximar una solución",
                "table": table,
                "is_successful": False,
                "have_solution": False,
                "solution": [],
                "spectral_radius": spectral_radius,
            }

        # Si la matriz es 2x2, generar las gráficas
        if len(A) == 2:
            plot_matrix_solution(table, x.tolist(), spectral_radius)
            plot_system_equations(A.tolist(), b.tolist(), x.tolist())

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
        relaxation_factor: float,
        matrix_size: int,
        **kwargs,
    ) -> str | list:

        # Validación de los parámetros de entrada tolerancia positiva
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            return "La tolerancia debe ser un número positivo"

        # Validación de los parámetros de entrada máximo número de iteraciones positivo
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

        # Validar que A es de máximo tamaño 6x6
        if len(A) > 6:
            return "La matriz A debe ser de hasta 6x6."

        # Validar que b y x0 tengan tamaños compatibles con A
        if len(b) != len(A) or len(x0) != len(A):
            return "El vector b y x0 deben ser compatibles con el tamaño de la matriz A."

        # Validar el rango del factor de relajación w
        if relaxation_factor <= 0 or relaxation_factor >= 2:
            return "El factor de relajación w debe estar en el rango (0, 2)."

        return [A, b, x0]
