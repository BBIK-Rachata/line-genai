import requests
import os

# Function to upload an image to the given URL
def upload_image(image_path: str, url: str):
    with open(image_path, 'rb') as img_file:
        files = {'file': img_file}
        response = requests.post(url, files=files)
        return response

if __name__ == "__main__":
    # Define the URL of the server where main.py is running
    server_url = "http://127.0.0.1:8080/detection"

    # Specify the path of the image file to upload
    image_path = r"C:\Users\SunanthineePiyarat\Desktop\SRISAWAD_detection\test_picture\demage.jpg"  # Change to the path of the image you want to upload


    # Check if the file exists
    if os.path.exists(image_path):
        response = upload_image(image_path, server_url)
        
        # Check the result of the upload
        if response.status_code == 200:
            print("Image uploaded successfully!")
            # If you want to save the result (such as the image from detection)
            with open("result_image.png", "wb") as f:
                f.write(response.content)
            print("Result image saved as result_image.png")
        else:
            print(f"Failed to upload image. Status code: {response.status_code}")
    else:
        print(f"Image file {image_path} does not exist.")
