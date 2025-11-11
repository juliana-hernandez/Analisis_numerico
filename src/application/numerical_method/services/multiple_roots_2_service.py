import sympy as sp
import math
from src.application.numerical_method.interfaces.iterative_method import (
    IterativeMethod,
)
from src.application.shared.utils.convert_math_to_simply import convert_math_to_sympy
from src.application.shared.utils.plot_function import plot_function

class MultipleRoots2Service(IterativeMethod):
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
        f_prime_expr = sp.diff(f_expr, x)  # Primera derivada
        f_double_prime_expr = sp.diff(f_prime_expr, x)  # Segunda derivada

        # Crea funciones evaluables en Python utilizando lambdify
        f = sp.lambdify(x, f_expr, modules=["math"])
        f_prime = sp.lambdify(x, f_prime_expr, modules=["math"])
        f_double_prime = sp.lambdify(x, f_double_prime_expr, modules=["math"])

        # Definición de tabla que contiene todo el proceso
        table = {}

        # Inicializa el valor inicial y error actual
        x0_current = x0
        current_error = math.inf
        current_iteration = 1

        # Bucle del método de raíces múltiples con segunda derivada
        while current_iteration <= max_iterations:
            try:
                fx = f(x0_current)
                f_prime_x = f_prime(x0_current)
                f_double_prime_x = f_double_prime(x0_current)

                if f_prime_x == 0 or f_double_prime_x == 0:
                    return {
                        "message_method": f"Una de las derivadas es cero en x = {x0_current}. No se puede continuar.",
                        "table": table,
                        "is_successful": True,
                        "have_solution": False,
                        "root": 0.0,
                    }

                # Aplicar la fórmula del método de raíces múltiples con segunda derivada
                x_next = x0_current - (fx * f_prime_x) / (
                    f_prime_x**2 - fx * f_double_prime_x
                )
            except Exception as e:
                return {
                    "message_method": f"Error al evaluar la función o sus derivadas: {str(e)}.",
                    "table": {},
                    "is_successful": True,
                    "have_solution": False,
                    "root": 0.0,
                }

            # Guardar los datos de la iteración actual en la tabla
            table[current_iteration] = {
                "iteration": current_iteration,
                "approximate_value": x0_current,
                "f_evaluated": fx,
                "f_prime_evaluated": f_prime_x,
                "f_double_prime_evaluated": f_double_prime_x,
                "next_x": x_next,
            }

            # Calcular el error dependiendo de la precisión
            if current_iteration > 1:
                if precision:  # Error absoluto
                    current_error = abs(x_next - x0_current)
                else:  # Error relativo
                    current_error = abs((x_next - x0_current) / x_next)
                table[current_iteration]["error"] = current_error
            else:
                table[current_iteration]["error"] = current_error  # Primera iteración

            # Verificar si se ha encontrado una raíz exacta o una aproximación aceptable
            if fx == 0:
                return {
                    "message_method": f"{x0_current} es una raíz exacta de f(x).",
                    "table": table,
                    "is_successful": True,
                    "have_solution": True,
                    "root": x0_current,
                }
            if current_iteration > 1 and current_error < tolerance:
                return {
                    "message_method": f"{x0_current} es una aproximación de la raíz de f(x) con un error menor a {tolerance}.",
                    "table": table,
                    "is_successful": True,
                    "have_solution": True,
                    "root": x0_current,
                }

            # Actualizar x0 para la siguiente iteración
            x0_current = x_next
            current_iteration += 1

        # Si se alcanzó el número máximo de iteraciones sin encontrar una raíz
        return {
            "message_method": f"El método funcionó correctamente pero no se encontró solución en {max_iterations} iteraciones.",
            "table": table,
            "is_successful": True,
            "have_solution": False,
            "root": 0.0,
        }

    def validate_input(
        self,
        x0: float,
        tolerance: float,
        max_iterations: int,
        function_f: str,
        **kwargs,
    ) -> str | bool:
        # Inicializa la variable simbólica para usar en SymPy
        x = sp.symbols("x")
        # Convierte la función ingresada de `math` a `SymPy`
        sympy_function_f = convert_math_to_sympy(function_f)

        # Validación de los parámetros de entrada tolerancia positiva
        if not isinstance(tolerance, (int, float)) or tolerance <= 0:
            plot_function(function_f, False, [(x0, 0)])
            return "La tolerancia debe ser un número positivo"

        # Validación de los parámetros de entrada máximo número de iteraciones positivo
        if not isinstance(max_iterations, int) or max_iterations <= 0:
            plot_function(function_f, False, [(x0, 0)])
            return "El máximo número de iteraciones debe ser un entero positivo."

        try:
            f_expr = sp.sympify(sympy_function_f)
            if f_expr.free_symbols != {x}:
                return "Error al interpretar la función: utilice la variable 'x'."
            f_prime_expr = sp.diff(f_expr, x)
            f_double_prime_expr = sp.diff(f_prime_expr, x)
        except Exception as e:
            return f"Error al interpretar o derivar la función ingresada: {str(e)}."

        try:
            sp.lambdify(x, f_expr, modules=["math"])
            sp.lambdify(x, f_prime_expr, modules=["math"])
            sp.lambdify(x, f_double_prime_expr, modules=["math"])
        except Exception as e:
            return f"Error al convertir la función o sus derivadas a formato numérico: {str(e)}."

        return True
