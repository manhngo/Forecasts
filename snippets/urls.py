from django.urls import path, include
from .views import *

urlpatterns = [
    path('forecast', get_forecast),
]