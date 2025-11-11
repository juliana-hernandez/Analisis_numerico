import numpy as np
from config.settings import BASE_DIR
import matplotlib.pyplot as plt
import math
import textwrap
import matplotlib

matplotlib.use("Agg")


def plot_function(
    function_f: str, have_solution: bool, points: list[tuple[float, float]]
) -> None:
    output_file = BASE_DIR / "static/img/numerical_method/function_plot.svg"

    # Obtener los valores de x y y de los puntos para el rango de la gráfica
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    min_x, max_x = min(x_coords) - 1, max(x_coords) + 1
    min_y, max_y = min(y_coords) - 1, max(y_coords) + 1

    # Generar un rango más ajustado de valores de x para la evaluación de la función
    if not have_solution:
        x_vals = np.linspace(-100, 100, 10000)
    else:
        x_vals = np.linspace(min_x - 3, max_x + 3, 10000)

    # Evaluar la función de forma segura
    y_vals = []
    valid_x = []
    for val in x_vals:
        try:
            y = eval(function_f, {"math": math, "x": val})
            if not (np.isnan(y) or np.isinf(y)):
                y_vals.append(y)
                valid_x.append(val)
        except Exception:
            continue

    # Crear la figura
    plt.figure(figsize=(6, 4))

    # Graficar la función en el rango ajustado
    plt.plot(valid_x, y_vals, color="#a18262", label="f(x)")

    if have_solution:
        for x, y in points:
            plt.plot(x, y, marker="o", color="#f7dc6f")
            plt.text(x, y, f"({x}, {y})", fontsize=9, verticalalignment="bottom")

    # Ajustar los límites de los ejes para el recuadro de la grafica como tal
    plt.xlim(min_x - 3, max_x + 3)
    plt.ylim(min_y - 3, max_y + 3)

    # Ejes y etiquetas
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")

    wrapped_title = "\n".join(textwrap.wrap(function_f, width=50))
    plt.title(f"f(x) = {wrapped_title}", fontsize=10, y=1.1)

    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Guardar la gráfica
    plt.savefig(output_file, format="svg")
    plt.close()
