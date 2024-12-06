import requests
from dotenv import load_dotenv
import os

load_dotenv()


url = "https://api.anthropic.com/v1/messages"
headers = {
    "content-type": "application/json",
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
}

payload = {
    "model": "claude-3-5-sonnet-20241022",
    "max_tokens": 1024,
    "tools": [
        {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    }
                },
                "required": ["location"],
            },
        }
    ],
    "messages": [
        {"role": "user", "content": "What is the weather like in San Francisco?"}
    ],
}

response = requests.post(url, json=payload, headers=headers)
print(response.json())
