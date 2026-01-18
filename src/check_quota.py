import requests
import json

def check_openrouter_quota(api_key):
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {
        "Authorization": f"Bearer {api_key}"
    }
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            print("API Key Information:")
            print(json.dumps(data, indent=2))
            
            # OpenRouter often returns 'limit' and 'usage' in the 'data' object
            if 'data' in data:
                limit = data['data'].get('limit')
                usage = data['data'].get('usage')
                if limit is not None and usage is not None:
                    remaining = limit - usage
                    print(f"\nCalculated Remaining Credits: {remaining}")
                else:
                    print("\nCould not calculate exact remaining credits from response fields (limit/usage might be null if unlimited or prepaid).")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"An error occurred: {e}")

def test_generation(api_key):
    print("\nTesting simple generation...")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "openai/gpt-3.5-turbo", # Use a cheap/standard model for testing
        "messages": [{"role": "user", "content": "Say hello!"}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print("Generation successful!")
            print(response.json()['choices'][0]['message']['content'])
        else:
            print(f"Generation failed: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Generation error: {e}")

if __name__ == "__main__":
    API_KEY = "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"
    check_openrouter_quota(API_KEY)
    # test_generation(API_KEY) # Optional: uncomment to test actual generation
