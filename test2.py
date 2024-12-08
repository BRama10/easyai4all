# from easyai4all.providers.options.openai import OpenAI
from easyai4all.client import OpenAI, Client

# client = OpenAI()
client = Client()

messages = [{"role": "user", "content": "Tell me a joke about programming"}]

response = client.create(model="ollama/llama3.2", messages=messages)

print(response.choices[0].message)
