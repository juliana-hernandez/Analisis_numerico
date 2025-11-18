from django.views.generic import TemplateView
from src.application.numerical_method.interfaces.interpolation_method import (
    InterpolationMethod,
)
from src.application.numerical_method.containers.numerical_method_container import (
    NumericalMethodContainer,
)
from dependency_injector.wiring import inject, Provide
from src.application.shared.utils.plot_function import plot_function
from django.http import HttpRequest, HttpResponse


class LagrangeView(TemplateView):
    template_name = "lagrange.html"

    @inject
    def __init__(
        self,
        method_service: InterpolationMethod = Provide[
            NumericalMethodContainer.lagrange_service
        ],
        **kwargs
    ):
        super().__init__(**kwargs)
        self.method_service = method_service

    def post(
        self, request: HttpRequest, *args: object, **kwargs: object
    ) -> HttpResponse:
        context = self.get_context_data()

        template_data = {}

        # Obtener entradas del formulario
        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        # Validar entradas
        response_validation = self.method_service.validate_input(x_input, y_input)

        if isinstance(response_validation, str):
            # Si hay un error de validación, agregarlo al contexto
            error_response = {
                "message_method": response_validation,
                "is_successful": False,
                "have_solution": False,
            }
            template_data = template_data | error_response
            context["template_data"] = template_data
            return self.render_to_response(context)

        # Si las entradas son válidas, extraer los valores
        x_values = response_validation[0]
        y_values = response_validation[1]

        # Ordenar puntos para la representación gráfica
        points = list(zip(x_values, y_values))
        sorted_points = sorted(points, key=lambda point: point[0])
        
        # Ejecutar el método de Lagrange
        show_error_report = (request.POST.get("show_error_report") == "on")
        method_response = self.method_service.solve(
            x=x_values,
            y=y_values,
            show_error_report=show_error_report,
        )


        if method_response["is_successful"]:
            # Graficar la función interpolante
            plot_function(
                method_response["polynomial"],
                method_response["have_solution"],
                sorted_points,
            )

        # Agregar respuesta del método al contexto
        template_data = template_data | method_response
        context["template_data"] = template_data

        return self.render_to_response(context)