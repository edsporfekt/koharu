import requests
import json

api_key = "AIzaSyCwHtNrCM6D-vv3q8RQHmVbTbi8DMpH41U"
model_id = "gemma-4-31b-it" # Assuming this is the correct ID from mod.rs
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

def test_gemma_thinking(use_think_token=True, use_sampling=True):
    system_prompt = "You are a helpful assistant."
    if use_think_token:
        system_prompt = "<|think|> " + system_prompt
    
    payload = {
        "system_instruction": {
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "parts": [{"text": "What is 123 * 456? Show your thinking."}]
            }
        ]
    }
    
    if use_sampling:
        payload["generationConfig"] = {
            "temperature": 1.0,
            "topP": 0.95,
            "topK": 64
        }
    
    print(f"--- Testing use_think_token={use_think_token}, use_sampling={use_sampling} ---")
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        try:
            text = result['candidates'][0]['content']['parts'][0]['text']
            print("Response text:")
            print(text)
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(json.dumps(result, indent=2))
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_gemma_thinking(use_think_token=True, use_sampling=True)
