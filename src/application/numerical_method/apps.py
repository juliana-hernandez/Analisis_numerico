from django.apps import AppConfig
from .containers.numerical_method_container import NumericalMethodContainer


class NumericalMethodConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "src.application.numerical_method"

    def ready(self) -> None:
        container = NumericalMethodContainer()
        container.wire(
            modules=[
                "src.application.numerical_method.views.bisection_view",
                "src.application.numerical_method.views.regula_falsi_view",
                "src.application.numerical_method.views.fixed_point_view",
                "src.application.numerical_method.views.newton_raphson_view",
                "src.application.numerical_method.views.secant_view",
                "src.application.numerical_method.views.multiple_roots_1_view",
                "src.application.numerical_method.views.multiple_roots_2_view",
                "src.application.numerical_method.views.jacobi_view",
                "src.application.numerical_method.views.gauss_seidel_view",
                "src.application.numerical_method.views.sor_view",
                "src.application.numerical_method.views.vandermonde_view",
                "src.application.numerical_method.views.spline_linear_view",
                "src.application.numerical_method.views.spline_cubic_view",
                "src.application.numerical_method.views.lagrange_view",
                "src.application.numerical_method.views.newton_interpol_view",
            ]
        )
