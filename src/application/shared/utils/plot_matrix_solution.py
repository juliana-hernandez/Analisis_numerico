import matplotlib.pyplot as plt
import numpy as np
from config.settings import BASE_DIR


def plot_matrix_solution(iterations: dict, solution: list[float], spectral_radius: float):
    """
    Grafica las soluciones iterativas de un sistema de ecuaciones lineales (Jacobi para matrices 2x2).

    Args:
        iterations (dict): Diccionario con las iteraciones y los valores aproximados de X.
        solution (list[float]): Solución aproximada del sistema.
        spectral_radius (float): Radio espectral para mostrar en la gráfica.

    Returns:
        None: Genera un archivo SVG con la gráfica.
    """
    output_file = BASE_DIR / "static/img/numerical_method/matrix_solution_plot.svg"

    # Extraer valores de iteración
    x1_values = [iteration["X"][0] for iteration in iterations.values()]
    x2_values = [iteration["X"][1] for iteration in iterations.values()]
    iteration_numbers = list(iterations.keys())

    # Crear la figura
    plt.figure(figsize=(8, 6))

    # Graficar las soluciones x1 y x2 por iteración
    plt.plot(iteration_numbers, x1_values, label="x1 (iterativo)", marker="o", linestyle="-")
    plt.plot(iteration_numbers, x2_values, label="x2 (iterativo)", marker="o", linestyle="-")

    # Añadir la solución final
    plt.axhline(y=solution[0], color="blue", linestyle="-", label=f"x1 solución: {solution[0]:.4f}")
    plt.axhline(y=solution[1], color="green", linestyle="-", label=f"x2 solución: {solution[1]:.4f}")

    # Detalles de la gráfica
    plt.title(f"Evolución iterativa (Radio espectral: {spectral_radius:.4f})")
    plt.xlabel("Iteraciones")
    plt.ylabel("Valor de X")
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica como SVG
    plt.savefig(output_file, format="svg")
    plt.close()


def plot_system_equations(A: list[list[float]], b: list[float], solution: list[float]):
    """
    Genera una gráfica de las ecuaciones de un sistema 2x2 y su solución.

    Args:
        A (list[list[float]]): Matriz de coeficientes (2x2).
        b (list[float]): Vector de términos independientes.
        solution (list[float]): Solución del sistema.

    Returns:
        None: Genera un archivo SVG con la gráfica.
    """
    output_file = BASE_DIR / "static/img/numerical_method/system_plot.svg"

    # Crear las ecuaciones como funciones de x
    x = np.linspace(-10, 10, 500)
    y1 = (b[0] - A[0][0] * x) / A[0][1]  # Primera ecuación
    y2 = (b[1] - A[1][0] * x) / A[1][1]  # Segunda ecuación

    # Crear la gráfica
    plt.figure(figsize=(8, 6))
    
    # Graficar las ecuaciones con líneas continuas
    plt.plot(x, y1, label="Ecuación 1", color="blue", linestyle="-", linewidth=1.5)
    plt.plot(x, y2, label="Ecuación 2", color="green", linestyle="-", linewidth=1.5)
    
    # Añadir el punto de solución y su anotación
    plt.scatter(solution[0], solution[1], color="red", label="Solución", zorder=5)
    plt.text(
        solution[0],
        solution[1],
        f"({solution[0]:.4f}, {solution[1]:.4f})",
        fontsize=10,
        verticalalignment="bottom",
        horizontalalignment="right",
    )

    # Detalles de la gráfica
    plt.title("Sistema de ecuaciones 2x2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.grid(True)
    plt.legend()

    # Guardar la gráfica como SVG
    plt.savefig(output_file, format="svg")
    plt.close()
