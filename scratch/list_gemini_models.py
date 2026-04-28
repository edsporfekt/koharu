import requests
import json

api_key = "AIzaSyCwHtNrCM6D-vv3q8RQHmVbTbi8DMpH41U"
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"

response = requests.get(url)
if response.status_code == 200:
    models = response.json().get("models", [])
    print("Available models:")
    for model in models:
        print(f"Name: {model['name']}, Display Name: {model['displayName']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
