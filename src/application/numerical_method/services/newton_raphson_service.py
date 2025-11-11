import sympy as sp
import math
from src.application.numerical_method.interfaces.iterative_method import (
    IterativeMethod,
)
from src.application.shared.utils.convert_math_to_simply import convert_math_to_sympy
from src.application.shared.utils.plot_function import plot_function


class NewtonService(IterativeMethod):

    def solve(
        self,
        x0: float,
        tolerance: float,
        max_iterations: int,
        precision: int,
        function_f: str,
        **kwargs,
    ) -> dict:
        # Inicializa la variable simbólica para usar en SymPy
        x = sp.symbols("x")
        # Convierte la función ingresada de `math` a `SymPy`
        sympy_function_f = convert_math_to_sympy(function_f)
        # Intenta convertir la función a una expresión simbólica usando sympify
        f_expr = sp.sympify(sympy_function_f)
        f_prime_expr = sp.diff(f_expr, x)  # Calcular la derivada simbólicamente
        # Crea funciones evaluables en Python utilizando lambdify
        f = sp.lambdify(x, f_expr, modules=["math"])
        f_prime = sp.lambdify(x, f_prime_expr, modules=["math"])

        # Definición de tabla que contiene todo el proceso
        table = {}
        # Inicializa el valor inicial y error actual
        x0_current = x0
        current_error = math.inf
        current_iteration = 1
        points = [(x0_current, 0)]  # Para graficar

        while current_iteration <= max_iterations:
            # Evaluar f(x) y f'(x) en el valor actual de x0
            try:
                fx = f(x0_current)
                f_prime_x = f_prime(x0_current)
                if f_prime_x == 0:
                    return self._prepare_response(
                        message=f"La derivada es cero en x = {x0_current}. No se puede continuar.",
                        table=table,
                        is_successful=True,
                        have_solution=False,
                        points=points,
                        function=function_f,
                    )
                # Calcular el siguiente valor de x usando el método
                x_next = x0_current - fx / f_prime_x
            except Exception as e:
                return self._prepare_response(
                    message=f"Error al evaluar la función o su derivada: {str(e)}.",
                    table=table,
                    is_successful=True,
                    have_solution=False,
                    points=points,
                    function=function_f,
                )

            # Guardar los datos de la iteración actual en la tabla
            error_value = (
                abs(x_next - x0_current)  # Error absoluto
                if precision  # Condición para error absoluto
                else abs((x_next - x0_current) / x_next)  # Error relativo
            )

            table[current_iteration] = {
                "iteration": current_iteration,
                "approximate_value": x0_current,
                "f_evaluated": fx,
                "f_prime_evaluated": f_prime_x,
                "next_x": x_next,
                "error": error_value,
            }
            points.append((x0_current, fx))  # Agregar puntos para graficar

            # Verificar si se ha encontrado una raíz exacta o una aproximación aceptable
            if fx == 0 or error_value < tolerance:
                return self._prepare_response(
                    message=f"{x0_current} es una aproximación de la raíz de f(x) con un error menor a {tolerance}.",
                    table=table,
                    is_successful=True,
                    have_solution=True,
                    points=points,
                    function=function_f,
                )

            # Actualizar x0 para la siguiente iteración
            x0_current = x_next
            current_iteration += 1

        # Si se alcanzó el número máximo de iteraciones sin encontrar una raíz
        return self._prepare_response(
            message=f"El método funcionó correctamente pero no se encontró solución en {max_iterations} iteraciones.",
            table=table,
            is_successful=False,
            have_solution=False,
            points=points,
            function=function_f,
        )

    def _prepare_response(
        self,
        message: str,
        table: dict,
        is_successful: bool,
        have_solution: bool,
        points: list,
        function: str,
    ) -> dict:
        """Prepara la respuesta y genera la gráfica de la función."""
        plot_function(
            function_f=function,
            have_solution=have_solution,
            points=points,
        )
        return {
            "message_method": message,
            "table": table,
            "is_successful": is_successful,
            "have_solution": have_solution,
            "root": points[-1][0] if have_solution else 0.0,
        }

    def validate_input(
        self,
        x0: float,
        tolerance: float,
        max_iterations: int,
        function_f: str,
        **kwargs,
    ) -> str | bool:
        # Validaciones
        x = sp.symbols("x")
        sympy_function_f = convert_math_to_sympy(function_f)

        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            plot_function(function_f, False, [(x0, 0)])
            return "La tolerancia debe ser un número positivo"

        if not isinstance(max_iterations, int) or max_iterations <= 0:
            plot_function(function_f, False, [(x0, 0)])
            return "El máximo número de iteraciones debe ser un entero positivo."

        try:
            f_expr = sp.sympify(sympy_function_f)
            if f_expr.free_symbols != {x}:
                return "Error al interpretar la función: utilice la variable 'x'."
            sp.diff(f_expr, x)
        except Exception as e:
            return f"Error al interpretar la función ingresada: {str(e)}."

        return True
