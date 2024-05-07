import requests
import os
import random

def get_image_links():
    url = "https://api.github.com/repos/EliSchwartz/imagenet-sample-images/contents/"
    headers = {'Accept': 'application/vnd.github.v3+json'}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        image_links = []
        for item in response.json():
            if item['name'].endswith('.JPEG'):
                image_links.append(item['download_url'])  # 
        return image_links
    else:
        print(f"Failed to fetch data: {response.status_code}")
        return []

def download_images(image_links):
    selected_images = random.sample(image_links, min(50, len(image_links)))
    os.makedirs('pictures', exist_ok=True)

    for idx, image_url in enumerate(selected_images):
        response = requests.get(image_url)
        if response.status_code == 200:
            with open(f'pictures/image_{idx+1}.JPEG', 'wb') as f:
                f.write(response.content)
            print(f"Downloaded {image_url}")
        else:
            print(f"Failed to download {image_url}")

if __name__ == "__main__":
    image_links = get_image_links()
    if image_links:
        print(f"Found {len(image_links)} images, downloading up to 50 of them.")
        download_images(image_links)
    else:
        print("No images found in the repository.")







