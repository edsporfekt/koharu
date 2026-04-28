import requests
import json

api_key = "AIzaSyCwHtNrCM6D-vv3q8RQHmVbTbi8DMpH41U"
model_id = "gemma-4-31b-it"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

def test_gemma_thinking():
    # Use EXACTLY what the user suggested
    system_prompt = "<|think|> You are a professional manga translator."
    
    payload = {
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "parts": [{"text": "[1] Hello, how are you today?"}]
            }
        ],
        "generationConfig": {
            "temperature": 1.0,
            "topP": 0.95,
            "topK": 64
        }
    }
    
    print(f"--- Testing Gemma 4 Thinking ---")
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_gemma_thinking()
