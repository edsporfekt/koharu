import requests
import json
import sys
import io

# Force UTF-8 output for Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

api_key = "AIzaSyCwHtNrCM6D-vv3q8RQHmVbTbi8DMpH41U"
model_id = "gemma-4-31b-it"
url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"

BLOCK_TAG_INSTRUCTIONS = "The input uses numbered tags like [1], [2], etc. to mark each text block. Translate only the text after each tag. Keep every tag exactly unchanged, including numbers and order. Output the same tags followed by the translated text. Do not merge, split, or reorder blocks."

def system_prompt(target_language):
    return f"You are a professional manga translator. Translate manga dialogue into natural {target_language} that fits inside speech bubbles. Preserve character voice, emotional tone, relationship nuance, emphasis, and sound effects naturally. Keep the wording concise. Do not add notes, explanations, or romanization. {BLOCK_TAG_INSTRUCTIONS}"

def test_gemma_with_koharu_prompt():
    target_lang = "Turkish"
    # To trigger thinking as per user instructions
    full_system_prompt = "<|think|> " + system_prompt(target_lang)
    
    sample_text = "[1] お前、何者だ？\n[2] 私はただの通りすがりの仮面ライダーだ！\n[3] 覚えておけ！"
    
    payload = {
        "systemInstruction": {
            "parts": [{"text": full_system_prompt}]
        },
        "contents": [
            {
                "parts": [{"text": sample_text}]
            }
        ],
        "generationConfig": {
            "temperature": 1.0,
            "topP": 0.95,
            "topK": 64
        }
    }
    
    print(f"--- Testing Gemma 4 with Koharu Prompt (Thinking Enabled) ---")
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        result = response.json()
        
        if 'candidates' not in result:
             print(json.dumps(result, indent=2))
             return

        parts = result['candidates'][0]['content']['parts']
        
        print("\n--- Response Parts ---")
        for i, part in enumerate(parts):
            is_thought = part.get("thought", False)
            print(f"Part {i} ({'THOUGHT' if is_thought else 'TEXT'}):")
            print(part.get("text", ""))
            print("-" * 20)
            
        # Final result (what my code now returns)
        text = "".join([p.get("text", "") for p in parts if not p.get("thought", False)])
        print("\n--- Final Extracted Translation ---")
        print(text)
        
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_gemma_with_koharu_prompt()
