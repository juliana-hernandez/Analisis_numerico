from django.urls import path
from .views.index_view import Index

app_name = "home"
urlpatterns = [
    path("", Index.as_view(), name="index"),
]
