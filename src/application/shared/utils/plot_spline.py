import numpy as np
from config.settings import BASE_DIR
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import CubicSpline

matplotlib.use("Agg")


def plot_spline_linear(points: list[tuple[float, float]]) -> None:
    """
    Genera una gráfica para el spline lineal conectando los puntos dados.

    Args:
        points (list[tuple[float, float]]): Lista de puntos (x, y) para graficar.

    Returns:
        None: Genera un archivo SVG con la gráfica.
    """
    output_file = BASE_DIR / "static/img/numerical_method/spline_linear_plot.svg"

    # Crear la figura
    plt.figure(figsize=(6, 4))

    # Extraer coordenadas x e y de los puntos
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # Graficar las líneas entre los puntos
    for i in range(len(points) - 1):
        plt.plot(
            [x_coords[i], x_coords[i + 1]],
            [y_coords[i], y_coords[i + 1]],
            color="#a18262",
            label=f"Tramo {i + 1}" if i == 0 else "",
        )

    # Graficar los puntos individuales
    for x, y in points:
        plt.plot(x, y, marker="o", color="#f7dc6f")
        plt.text(x, y, f"({x:.1f}, {y:.1f})", fontsize=9, verticalalignment="bottom")

    # Ajustar los límites de los ejes
    min_x, max_x = min(x_coords) - 1, max(x_coords) + 1
    min_y, max_y = min(y_coords) - 1, max(y_coords) + 1
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    # Ejes y etiquetas
    plt.axhline(y=0, color="red", linestyle="--", linewidth=1)
    plt.axvline(x=0, color="red", linestyle="--", linewidth=1)
    plt.xlabel("x")
    plt.ylabel("y")

    plt.title("Spline Lineal", fontsize=12)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Guardar la gráfica
    plt.savefig(output_file, format="svg")
    plt.close()


def plot_spline_cubic(title: str, points: list[tuple[float, float]], x_values, y_values):
    output_file = BASE_DIR / "static/img/numerical_method/spline_cubic_plot.svg"

    plt.figure(figsize=(8, 6))

    # Crear el spline cúbico con scipy
    cs = CubicSpline(x_values, y_values, bc_type="natural")

    # Generar un rango continuo de x para graficar el spline cúbico
    x_range = np.linspace(min(x_values), max(x_values), 500)
    y_range = cs(x_range)

    # Graficar el spline cúbico
    plt.plot(x_range, y_range, label="Spline Cúbico", color="blue")

    # Graficar los puntos originales y etiquetarlos
    for x, y in points:
        plt.scatter(x, y, color="red")
        plt.text(
            x,
            y,
            f"({x:.1f}, {y:.1f})",
            fontsize=9,
            verticalalignment="bottom",
            horizontalalignment="right",
            color="black",
        )

    # Configuración de la gráfica
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.axvline(0, color="black", linewidth=0.5, linestyle="--")
    plt.legend()
    plt.grid(True)

    # Guardar la gráfica
    plt.savefig(output_file, format="svg")
    plt.close()
