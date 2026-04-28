import requests
import json

api_key = "AIzaSyCwHtNrCM6D-vv3q8RQHmVbTbi8DMpH41U"
models_to_test = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemma-4-31b-it",
    "gemma-4-26b-a4b-it"
]

def test_model(model_id):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": "Hello, respond with a very short greeting."}]
        }]
    }
    
    print(f"Testing model: {model_id}...", end=" ", flush=True)
    try:
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            print("SUCCESS")
            # print(response.json()["candidates"][0]["content"]["parts"][0]["text"])
        else:
            print(f"FAILED ({response.status_code})")
            print(response.text)
    except Exception as e:
        print(f"ERROR: {str(e)}")

for model in models_to_test:
    test_model(model)
