from easyai4all.providers.options.openai import OpenAI

client = OpenAI()

messages = [
        {
            "role": "user",
            "content": "Tell me a joke about programming"
        }
    ]

response = client.create(model='gpt-4o', messages=messages)

print(response.choices[0].message)