# from easyai4all.providers.options.openai import OpenAI
from easyai4all.client import OpenAI, Client
from easyai4all.chat_completion import ChatCompletionResponse

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

# assert here that completion is of type ChatCompletionResponse

print(completion.to_dict())

print(completion.choices[0].message.tool_calls)
