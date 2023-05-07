from django.urls import path
from .views import index, get_dog, get_cat, upload_image

app_name = 'page'
urlpatterns = [
    path('', index, name='index'),
    path('get_dog/', get_dog, name='get_dog'),
    path('get_cat/', get_cat, name='get_cat'),
    path('upload_image/', upload_image, name='upload_image'),
]    
   