# from easyai4all.providers.options.openai import OpenAI
from easyai4all.client import OpenAI, Client

# client = OpenAI()
client = Client()


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "unit": {"type": "string", "enum": ["c", "f"]},
                },
                "required": ["location", "unit"],
                "additionalProperties": False,
            },
        },
    }
]

completion = client.create(
    model="openai/gpt-4o",
    messages=[{"role": "user", "content": "What's the weather like in Paris today?"}],
    tools=tools,
)

print(completion.choices[0].message.tool_calls)
