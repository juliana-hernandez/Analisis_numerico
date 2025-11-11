from django.urls import path, include

urlpatterns = [
    path("", include("src.application.home.urls")),
    path("numerical-methods/", include("src.application.numerical_method.urls")),
]
