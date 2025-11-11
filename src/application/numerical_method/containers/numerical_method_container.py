from dependency_injector import containers, providers
from src.application.numerical_method.services.bisection_service import BisectionService
from src.application.numerical_method.services.regula_falsi_service import (
    RegulaFalsiService,
)
from src.application.numerical_method.services.fixed_point_service import (
    FixedPointService,
)
from src.application.numerical_method.services.fixed_point_service import (
    FixedPointService,
)
from src.application.numerical_method.services.newton_raphson_service import (
    NewtonService,
)
from src.application.numerical_method.services.secant_service import (
    SecantService,
)
from src.application.numerical_method.services.multiple_roots_1_service import (
    MultipleRoots1Service,
)
from src.application.numerical_method.services.multiple_roots_2_service import (
    MultipleRoots2Service,
)
from src.application.numerical_method.services.jacobi_service import JacobiService

from src.application.numerical_method.services.gauss_seidel_service import (
    GaussSeidelService,
)
from src.application.numerical_method.services.sor_service import (
    SORService,
)
from src.application.numerical_method.services.vandermonde_service import (
    VandermondeService,
)
from src.application.numerical_method.services.spline_linear_service import (
    SplineLinearService,
)
from src.application.numerical_method.services.spline_cubic_service import (
    SplineCubicService,
)
from src.application.numerical_method.services.lagrange_service import (
  LagrangeService,
)
from src.application.numerical_method.services.newton_interpol_service import (
  NewtonInterpolService,
)


class NumericalMethodContainer(containers.DeclarativeContainer):
    bisection_service = providers.Factory(BisectionService)
    regula_falsi_service = providers.Factory(RegulaFalsiService)
    fixed_point_service = providers.Factory(FixedPointService)
    newton_service = providers.Factory(NewtonService)
    secant_service = providers.Factory(SecantService)
    multiple_roots_1_service = providers.Factory(MultipleRoots1Service)
    multiple_roots_2_service = providers.Factory(MultipleRoots2Service)
    jacobi_service = providers.Factory(JacobiService)
    gauss_seidel_service = providers.Factory(GaussSeidelService)
    sor_service = providers.Factory(SORService)
    vandermonde_service = providers.Factory(VandermondeService)
    spline_linear_service = providers.Factory(SplineLinearService)
    spline_cubic_service = providers.Factory(SplineCubicService)
    lagrange_service = providers.Factory(LagrangeService)
    newton_interpol_service = providers.Factory(NewtonInterpolService)
