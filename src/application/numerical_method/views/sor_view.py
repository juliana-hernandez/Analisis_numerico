from django.views.generic import TemplateView
from src.application.numerical_method.interfaces.matrix_method import MatrixMethod
from src.application.numerical_method.containers.numerical_method_container import (
    NumericalMethodContainer,
)
from dependency_injector.wiring import inject
from django.http import HttpRequest, HttpResponse


class SORView(TemplateView):
    template_name = "sor.html"

    @inject
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.method_service = NumericalMethodContainer.sor_service()

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        # Agregando los tamaños de matriz al contexto
        context["matrix_sizes"] = [2, 3, 4, 5, 6]
        return context

    def post(
        self, request: HttpRequest, *args: object, **kwargs: object
    ) -> HttpResponse:
        context = self.get_context_data()

        template_data = {}

        # Obtener datos del formulario
        matrix_a_raw = request.POST.get("matrix_a", "")
        vector_b_raw = request.POST.get("vector_b", "")
        initial_guess_raw = request.POST.get("initial_guess", "")
        tolerance = float(request.POST.get("tolerance"))
        max_iterations = int(request.POST.get("max_iterations"))
        relaxation_factor = float(request.POST.get("relaxation_factor"))
        precision_type = int(request.POST.get("precision"))
        matrix_size = int(request.POST.get("matrix_size"))

        # Validar entrada
        response_validation = self.method_service.validate_input(
            matrix_a_raw=matrix_a_raw,
            vector_b_raw=vector_b_raw,
            initial_guess_raw=initial_guess_raw,
            tolerance=tolerance,
            max_iterations=max_iterations,
            relaxation_factor=relaxation_factor,
            matrix_size=matrix_size,
        )

        if isinstance(response_validation, str):
            # Si hay errores de validación, devolver mensaje de error
            error_response = {
                "message_method": response_validation,
                "table": {},
                "is_successful": False,
                "have_solution": False,
                "solution": [],
            }
            template_data = template_data | error_response
            context["template_data"] = template_data
            return self.render_to_response(context)

        # Obtener los valores de A, b y x0
        A = response_validation[0]
        b = response_validation[1]
        x0 = response_validation[2]

        # Ejecutar el método SOR con los parámetros recibidos
        method_response = self.method_service.solve(
            A=A,
            b=b,
            x0=x0,
            tolerance=tolerance,
            max_iterations=max_iterations,
            relaxation_factor=relaxation_factor,
            precision_type=precision_type,  # Nuevo argumento para precisión
            show_error_report=(request.POST.get("show_error_report") == "on"),
        )

        # Verificación de éxito y almacenamiento de la respuesta
        template_data["indexes"] = list(range(1, len(A) + 1))
        template_data["relaxation_factor"] = relaxation_factor
        template_data = template_data | method_response
        context["template_data"] = template_data
        return self.render_to_response(context)
