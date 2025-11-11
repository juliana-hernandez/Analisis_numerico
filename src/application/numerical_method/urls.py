from django.urls import path
from .views.file_download_view import FileDownloadView
from .views.bisection_view import BisectionView
from .views.regula_falsi_view import RegulaFalsiView
from .views.fixed_point_view import FixedPointView
from .views.newton_raphson_view import NewtonRaphsonView
from .views.secant_view import SecantView
from .views.multiple_roots_1_view import MultipleRoots1View
from .views.multiple_roots_2_view import MultipleRoots2View
from .views.jacobi_view import JacobiView
from .views.gauss_seidel_view import GaussSeidelView
from .views.sor_view import SORView
from .views.vandermonde_view import VandermondeView
from .views.spline_linear_view import SplineLinearView
from .views.spline_cubic_view import SplineCubicView
from .views.lagrange_view import LagrangeView
from .views.newton_interpol_view import NewtonInterpolView

app_name = "numerical_method"
urlpatterns = [
    path(
        "download-svg/",
        FileDownloadView.as_view(),
        name="download_svg",
    ),
    path(
        "bisection/",
        BisectionView.as_view(),
        name="bisection",
    ),
    path(
        "regula-falsi/",
        RegulaFalsiView.as_view(),
        name="regula_falsi",
    ),
    path(
        "fixed-point/",
        FixedPointView.as_view(),
        name="fixed_point",
    ),
    path(
        "newton_raphson/",
        NewtonRaphsonView.as_view(),
        name="newton_raphson",
    ),
    path(
        "secant/",
        SecantView.as_view(),
        name="secant",
    ),
    path(
        "multiple_roots_1/",
        MultipleRoots1View.as_view(),
        name="multiple_roots_1",
    ),
    path(
        "multiple_roots_2/",
        MultipleRoots2View.as_view(),
        name="multiple_roots_2",
    ),
    path(
        "jacobi/",
        JacobiView.as_view(),
        name="jacobi",
    ),
    path(
        "gauss-seidel/",
        GaussSeidelView.as_view(),
        name="gauss_seidel",
    ),
    path(
        "sor/",
        SORView.as_view(),
        name="sor",
    ),
    path(
        "vandermonde/",
        VandermondeView.as_view(),
        name="vandermonde",
    ),
    path(
        "spline-linear/",
        SplineLinearView.as_view(),
        name="spline_linear",
    ),
    path(
        "spline-cubic/",
        SplineCubicView.as_view(),
        name="spline_cubic",
    ),
    path(
        "lagrange/",
        LagrangeView.as_view(),
        name="lagrange",
    ),
    path(
        "newton-interpol/",
        NewtonInterpolView.as_view(),
        name="newton-interpol",
    ),
]
