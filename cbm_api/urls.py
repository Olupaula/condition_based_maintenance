from django.urls import path
from . import views

urlpatterns = [
    path('', views.cbm_api, name='cbm_api')
]
