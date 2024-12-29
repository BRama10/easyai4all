# OpenAI

To use OpenAI with `easyai4all`, you’ll need an [OpenAI account](https://platform.openai.com/). After logging in, go to the [API Keys](https://platform.openai.com/account/api-keys) section in your account settings and generate a new key. Once you have your key, add it to your environment as follows:

```shell
export OPENAI_API_KEY="your-openai-api-key"
```

## Create a Chat Completion

In your code:
```python
from easyai4all.client import Client

client = Client()

provider = "openai"
model = "gpt-4-turbo"

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What’s the weather like in San Francisco?"},
]

response = client.create(
    model=f"{provider}/{model}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you’d like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).