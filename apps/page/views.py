import base64
from django.http import JsonResponse
import requests
from django.shortcuts import render
import json
import requests
from PIL import Image
from io import BytesIO
import tensorflow as tf
import numpy as np
from django.http import HttpResponseBadRequest

IMG_HEIGHT = 150
IMG_WIDTH = 150

    
def get_dog(request):
    """
    Fetches a random cat image URL from 'https://dog.ceo/api/breeds/image/random',
    uses a pre-trained model to predict if the image contains a dog or a cat, and returns the
    URL and the predicted label as a JSON response.
    """
    try:
        # Send GET request to the API endpoint
        response = requests.get('https://dog.ceo/api/breeds/image/random')

        # Check if the response is successful (status code 200)
        if response.status_code == 200:
            # Extract the JSON data from the response
            data = response.json()
            # Extract the 'url' key from the JSON data
            message = data['message']

            # Load the model
            model = tf.keras.models.load_model('cat_dog_classifier.h5')

            # Request the image from the URL and get the content
            response = requests.get(message)
            image_content = response.content

            # Decode the image data and create a PIL image object
            image = Image.open(BytesIO(image_content))
            # Resize the image to the expected size
            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            # Convert the PIL image to a numpy array
            image_array = np.array(image)
            # Normalize the image data
            image_array = image_array / 255.0
            # Return the 'message' and the predicted label as a JSON response
            # Add an extra dimension to the image array to match the input shape of the model
            image_array = np.expand_dims(image_array, axis=0)

            # Use the model to predict the class of the image
            prediction = model.predict(image_array)

            if prediction >= 0.5:
                label = 'dog'
            else:
                label = 'cat'

            return JsonResponse({'message': message, 'label': label})

    except Exception as e:
        print(e)
        return JsonResponse({'error': 'Failed to fetch cat image.'})
    
    
    
def get_cat(request):
    """
    Fetches a random cat image URL from 'https://api.thecatapi.com/v1/images/search?size=full',
    uses a pre-trained model to predict if the image contains a dog or a cat, and returns the
    URL and the predicted label as a JSON response.
    """
    try:
        # Send GET request to the API endpoint
        response = requests.get('https://api.thecatapi.com/v1/images/search?size=full')

        # Check if the response is successful (status code 200)
        if response.status_code == 200:
            # Extract the JSON data from the response
            data = response.json()
            # Extract the 'url' key from the JSON data
            url = data[0]['url']

            # Load the model
            model = tf.keras.models.load_model('cat_dog_classifier.h5')

            # Request the image from the URL and get the content
            response = requests.get(url)
            image_content = response.content

            # Decode the image data and create a PIL image object
            image = Image.open(BytesIO(image_content))
            # Resize the image to the expected size
            image = image.resize((IMG_WIDTH, IMG_HEIGHT))
            # Convert the PIL image to a numpy array
            image_array = np.array(image)
            # Normalize the image data
            image_array = image_array / 255.0
            # Return the 'url' and the predicted label as a JSON response
            # Add an extra dimension to the image array to match the input shape of the model
            image_array = np.expand_dims(image_array, axis=0)

            # Use the model to predict the class of the image
            prediction = model.predict(image_array)

            if prediction >= 0.5:
                label = 'dog'
            else:
                label = 'cat'

            return JsonResponse({'url': url, 'label': label})

    except Exception as e:
        print(e)
        return JsonResponse({'error': 'Failed to fetch cat image.'})
    
    # If the request is not POST or there is an exception, return a default response
    return HttpResponseBadRequest('Invalid request.')



def upload_image(request):
    if request.method == 'POST':
        print('post')
        # Load the model
        model = tf.keras.models.load_model('cat_dog_classifier.h5')

        # Get the image from the POST
        uploaded_file = request.FILES['image']
        # Decode the image data and create a PIL image object
        image = Image.open(BytesIO(uploaded_file.read()))
        # Convert the image to RGB color space (remove alpha channel)
        image = image.convert('RGB')
        # Resize the image to the expected size
        image = image.resize((IMG_WIDTH, IMG_HEIGHT))
        # Convert the PIL image to a numpy array
        image_array = np.array(image)
        # Normalize the image data
        image_array = image_array / 255.0
        # Add an extra dimension to the image array to match the input shape of the model
        image_array = np.expand_dims(image_array, axis=0)
        # Use the model to predict the class of the image
        prediction = model.predict(image_array)

        if prediction >= 0.5:
            label = 'dog'
        else:
            label = 'cat'

        return JsonResponse({'label': label})
    else:
        return HttpResponseBadRequest('Invalid request.')

def index(request):
    return render(request, 'pages/index.html')

