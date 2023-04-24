from flask import jsonify
import requests

def get_dog():
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
            return jsonify({'message': message})
        else:
            # If the response is unsuccessful, raise an exception
            response.raise_for_status()
    except Exception as e:
        # Handle any exceptions that may occur during the API request
        return jsonify({'error': str(e)}), 500