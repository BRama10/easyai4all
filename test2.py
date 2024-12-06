# from easyai4all.providers.options.openai import OpenAI
from easyai4all.client import OpenAI, Client

# client = OpenAI()
client = Client()

messages = [
        {
            "role": "user",
            "content": "Tell me a joke about programming"
        }
    ]

response = client.create(model='openai/gpt-4o', messages=messages)

print(response.choices[0].message)