import numpy as np
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
        x1 = np.zeros_like(x0)
        current_error = tolerance + 1
        current_iteration = 0
        table = {}

        # Inicialización de matrices para el cálculo de T y C
        D = np.diag(np.diag(A))
        L = np.tril(A, -1)
        U = np.triu(A, 1)

        # Cálculo de la matriz de iteración T para el método Jacobi
        T = np.linalg.inv(D).dot(L + U)
        spectral_radius = max(abs(np.linalg.eigvals(T)))

        while current_error > tolerance and current_iteration < max_iterations:
            # Iteración de Jacobi
            for i in range(n):
                sum_others = np.dot(A[i, :i], x0[:i]) + np.dot(
                    A[i, i + 1 :], x0[i + 1 :]
                )
                x1[i] = (b[i] - sum_others) / A[i, i]

            current_error = np.linalg.norm(x1 - x0, ord=np.inf)

            # Aplicar precisión según el tipo seleccionado
            formatted_x1 = self.apply_precision(x1.tolist(), precision_type, tolerance)
            formatted_error = self.apply_precision([current_error], precision_type, tolerance)[0]

            # Guardamos la información de la iteración actual
            table[current_iteration + 1] = {
                "iteration": current_iteration + 1,
                "X": formatted_x1,
                "Error": formatted_error,
            }

            # Preparación para la siguiente iteración
            x0 = x1.copy()
            current_iteration += 1

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
