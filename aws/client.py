import requests

# Define the URL of your Flask API endpoint
url = "http://localhost:5000/classify_image"

# Define the image file path
image_path = "imageduo.jpg"

# Create a dictionary with the file
files = {'file': open(image_path, 'rb')} 

# Send a POST request to the API endpoint
response = requests.post(url, files=files)

# Check the response
if response.status_code == 200:
    print("Image processed successfully:")
    print(response.json())
else:
    print("Error processing image. Status code:", response.status_code)