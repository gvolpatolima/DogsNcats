from django.urls import path
from .views import index, get_dog

urlpatterns = [
    path('', index, name='index'),
    path('get_dog/', get_dog, name='get_dog'),
]