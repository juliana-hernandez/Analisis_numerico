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


class NewtonInterpolView(TemplateView):
    template_name = "newton_interpol.html"

    @inject
    def __init__(
        self,
        method_service: InterpolationMethod = Provide[
            NumericalMethodContainer.newton_interpol_service
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
        x_extra_raw = request.POST.get("x_extra", "").strip()
        y_extra_raw = request.POST.get("y_extra", "").strip()

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
        points = list(zip(x_values, y_values))
        sorted_points = sorted(points, key=lambda point: point[0])

        # Parse optional extra point if provided
        x_extra = None
        y_extra = None
        if x_extra_raw != "" or y_extra_raw != "":
            try:
                if x_extra_raw != "":
                    x_extra = float(x_extra_raw)
                if y_extra_raw != "":
                    y_extra = float(y_extra_raw)
                # If only one provided, show validation error
                if (x_extra is None) != (y_extra is None):
                    error_response = {
                        "message_method": "Si ingresa dato adicional debe proporcionar ambos: x_{n+1} y y_{n+1}.",
                        "is_successful": False,
                        "have_solution": False,
                    }
                    template_data = template_data | error_response
                    context["template_data"] = template_data
                    return self.render_to_response(context)
            except ValueError:
                error_response = {
                    "message_method": "Los datos adicionales deben ser numéricos.",
                    "is_successful": False,
                    "have_solution": False,
                }
                template_data = template_data | error_response
                context["template_data"] = template_data
                return self.render_to_response(context)

        show_error_report = (request.POST.get("show_error_report") == "on")

        method_response = self.method_service.solve(
            x=x_values,
            y=y_values,
            x_extra=x_extra,
            y_extra=y_extra,
            show_error_report=show_error_report,
        )


        if method_response["is_successful"]:
            plot_function(
                method_response["polynomial"],
                method_response["have_solution"],
                sorted_points,
            )

        template_data = template_data | method_response
        context["template_data"] = template_data

        return self.render_to_response(context)
