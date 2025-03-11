import requests

API_URLS = {
    "vehicle": "http://127.0.0.1:8080/vehicle",
    "license": "http://127.0.0.1:8080/license"
}

IMAGE_PATH = r"C:\Users\SunanthineePiyarat\Desktop\SRISAWAD_detection\test_picture\BENZ.jpg"

def send_image(api_url, image_path):
    with open(image_path, "rb") as file:
        response = requests.post(api_url, files={"file": file})
        print(f"📌 Sending to: {api_url}")
        print("Response:", response.status_code)
        try:
            print("Result:", response.json())
        except Exception as e:
            print("Failed to parse JSON response", e)

if __name__ == "__main__":
    for api in API_URLS:
        send_image(API_URLS[api], IMAGE_PATH)
