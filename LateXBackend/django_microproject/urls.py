from django.urls import path
from django_microproject import views

urlpatterns = [
    path(r'api/internal/convert', views.convert, name='api')
]
