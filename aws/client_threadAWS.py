import os
import requests
import threading
import time

# Define the URL of your Flask API endpoint
url = "http://images-lb-1024619852.us-east-1.elb.amazonaws.com/classify_image"

def process_images(thread_id, folder_path, images):
    for image in images:
        image_path = os.path.join(folder_path, image)
        files = {'file': open(image_path, 'rb')}
        response = requests.post(url, files=files)

        if response.status_code == 200:
            print(f"Thread-{thread_id}: Image '{image}' processed successfully")
            print(response.json())
        else:
            print(f"Thread-{thread_id}: Error processing image '{image}'. Status code:", response.status_code)

def main():
    while True:
        # Define the folder containing the images
        pictures_folder = "pictures"

        # Check if the folder exists
        if not os.path.exists(pictures_folder):
            print(f"Folder '{pictures_folder}' not found.")
            return

        # Get list of images in the folder
        images = [filename for filename in os.listdir(pictures_folder) if filename.endswith((".JPEG", ".png",".JPG"))]

        print("Starting to parse the group of images")

        # Calculate number of images per thread
        num_images_per_thread = len(images) // 8

        # Create four threads
        threads = []
        for i in range(8):
            start_index = i * num_images_per_thread
            end_index = (i + 1) * num_images_per_thread if i < 7 else len(images)
            thread_images = images[start_index:end_index]
            thread = threading.Thread(target=process_images, args=(i+1, pictures_folder, thread_images))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        # Wait for some time before processing the images again
        print("All threads have been completed")
        print("Waiting 2 seconds before restarting the batch")
        time.sleep(2)  # Adjust this value as needed

if __name__ == "__main__":
    main()
