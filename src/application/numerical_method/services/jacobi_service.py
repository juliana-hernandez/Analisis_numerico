import numpy as np
import time
from src.application.numerical_method.interfaces.matrix_method import MatrixMethod
from src.application.shared.utils.plot_matrix_solution import plot_matrix_solution, plot_system_equations


class JacobiService(MatrixMethod):
    def solve(
        self,
        A: list[list[float]],  # Matriz de coeficientes
        b: list[float],  # Vector de términos independientes
        x0: list[float],  # Vector inicial de aproximación
        tolerance: float,  # Tolerancia para el error
        max_iterations: int,  # Número máximo de iteraciones
        precision_type: str = "decimales_correctos",  # Tipo de precisión
        **kwargs,
    ) -> dict:

        A = np.array(A)
        b = np.array(b)
        x0 = np.array(x0)

        n = len(b)
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

        # Cálculo de la matriz de iteración T para el método Jacobi
        T = np.linalg.inv(D).dot(L + U)
        spectral_radius = max(abs(np.linalg.eigvals(T)))

        # Iteraciones: usamos x_prev como el vector anterior y x1 la nueva aproximación
        x_prev = x0.copy()
        while current_iteration < max_iterations:
            iter_start = time.perf_counter()

            # Iteración de Jacobi
            for i in range(n):
                sum_others = np.dot(A[i, :i], x_prev[:i]) + np.dot(A[i, i + 1 :], x_prev[i + 1 :])
                x1[i] = (b[i] - sum_others) / A[i, i]

            # Calcular diferencias y métricas de error según especificado
            diff = x1 - x_prev

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

            # Aplicar precisión y guardar en la tabla principal
            formatted_x1 = self.apply_precision(x1.tolist(), precision_type, tolerance)
            formatted_error = self.apply_precision([float(np.nanmax(np.abs(diff)))], precision_type, tolerance)[0]

            table[current_iteration + 1] = {
                "iteration": current_iteration + 1,
                "X": formatted_x1,
                "Error": formatted_error,
            }

            # Construir informe de errores plano si el usuario lo pidió
            if show_error_report:
                error_entries.extend([
                    {"iteration": current_iteration + 1, "type": "Error absoluto", "value": val_abs, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 1", "value": val_rel1, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 2", "value": val_rel2, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 3", "value": val_rel3, "time_elapsed": iter_duration},
                    {"iteration": current_iteration + 1, "type": "Error relativo 4", "value": val_rel4, "time_elapsed": iter_duration},
                ])

            current_iteration += 1
            x_prev = x1.copy()

            # criterio de parada
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
                "solution": formatted_x1,
                "spectral_radius": spectral_radius,
            }
        elif current_iteration >= max_iterations:
            result = {
                "message_method": f"El método funcionó correctamente, pero no se encontró una solución en {max_iterations} iteraciones y el radio espectral es de = {spectral_radius}.",
                "table": table,
                "is_successful": True,
                "have_solution": False,
                "solution": formatted_x1,
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

        # Si la matriz es 2x2, generar la gráfica
        if len(A) == 2:
            plot_matrix_solution(table, x1.tolist(), spectral_radius)
            plot_system_equations(A.tolist(), b.tolist(), x1.tolist())

        if show_error_report:
            result["error_entries"] = error_entries

        return result

    def apply_precision(self, values, precision_type, tolerance):
        """
        Aplica precisión a una lista de valores basada en el tipo de precisión seleccionado.
        """
        if precision_type == "cifras_significativas":
            # Calcular cifras significativas según la tolerancia
            significant_figures = -int(np.floor(np.log10(tolerance)))
            return [round(value, significant_figures) for value in values]
        elif precision_type == "decimales_correctos":
            # Usar la cantidad de decimales basada en la tolerancia
            decimal_places = -int(np.floor(np.log10(tolerance)))
            return [round(value, decimal_places) for value in values]
        else:
            # Sin cambios si no se selecciona un tipo válido
            return values

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
