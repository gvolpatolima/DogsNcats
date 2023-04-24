from django.http import JsonResponse
import requests
from django.shortcuts import render

def get_dog(request):
    """
    Fetches a random dog image URL from 'https://dog.ceo/api/breeds/image/random' and returns as JSON response.
    """
    try:
        # Send GET request to the API endpoint
        response = requests.get('https://dog.ceo/api/breeds/image/random')

        # Check if the response is successful (status code 200)
        if response.status_code == 200:
            # Extract the JSON data from the response
            data = response.json()
            # Extract the 'message' key from the JSON data
            message = data['message']
            # Return the 'message' as JSON response
            return JsonResponse({'message': message})
        else:
            # If the response is unsuccessful, raise an exception
            response.raise_for_status()
    except Exception as e:
        # Handle any exceptions that may occur during the API request
        return JsonResponse({'error': str(e)}, status=500)

def index(request):
    return render(request, 'pages/index.html')
