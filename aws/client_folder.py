import os
import requests

print("code")

# Define the URL of your Flask API endpoint
url = "http://localhost:5000/classify_image"

def process_images(folder_path):
    print("pass the check")
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".JPEG") or filename.endswith(".png"):
            # Define the path of the image file
            image_path = os.path.join(folder_path, filename)
            # Create a dictionary with the file
            files = {'file': open(image_path, 'rb')}

            # Send a POST request to the API endpoint
            print("sent a request for" + image_path)
            response = requests.post(url, files=files)

            # Check the response
            if response.status_code == 200:
                print(f"Image '{filename}' processed successfully:")
                print(response.json())
            else:
                print(f"Error processing image '{filename}'. Status code:", response.status_code)

# Define the folder containing the images
pictures_folder = "pictures"

# Check if the folder exists
if os.path.exists(pictures_folder):
    process_images(pictures_folder)
else:
    print(f"Folder '{pictures_folder}' not found.")
