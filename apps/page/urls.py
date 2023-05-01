from django.urls import path
from .views import index, get_dog, get_cat

urlpatterns = [
    path('', index, name='index'),
    path('get_dog/', get_dog, name='get_dog'),
    path('get_cat/', get_cat, name='get_cat'),
]