from django.views.generic import TemplateView
from src.application.numerical_method.interfaces.interpolation_method import InterpolationMethod
from src.application.numerical_method.containers.numerical_method_container import (
    NumericalMethodContainer,
)
from dependency_injector.wiring import inject, Provide
from django.http import HttpRequest, HttpResponse


class SplineLinearView(TemplateView):
    template_name = "spline_linear.html"

    @inject
    def __init__(
        self,
        method_service: InterpolationMethod = Provide[
            NumericalMethodContainer.spline_linear_service
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

        x_input = request.POST.get("x", "")
        y_input = request.POST.get("y", "")

        response_validation = self.method_service.validate_input(x_input, y_input)

        if isinstance(response_validation, str):
            error_response = {
                "message_method": response_validation,
                "is_successful": False,
                "have_solution": False,
            }
            template_data = template_data | error_response
            context["template_data"] = template_data
            return self.render_to_response(context)

        x_values = response_validation[0]
        y_values = response_validation[1]

        method_response = self.method_service.solve(
            x=x_values,
            y=y_values,
        )

        if method_response["is_successful"]:
            # Incluir los tramos en el contexto si la solución fue exitosa
            template_data["tramos"] = method_response.get("tramos", [])

        template_data = template_data | method_response
        context["template_data"] = template_data

        return self.render_to_response(context)
