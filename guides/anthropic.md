# Anthropic

To use Anthropic with `easyai4all` you will need to [create an account](https://console.anthropic.com/login). Once logged in, go to the [API Keys](https://console.anthropic.com/settings/keys)
and click the "Create Key" button and export that key into your environment.


```shell
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Create a Chat Completion

In your code:
```python
from easyai4all.client import Client

client = Client()

provider = "anthropic"
model = "claude-3-5-sonnet-20241022"

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

response = client.create(
    model=f"{provider}/{model}",
    messages=messages,
)

print(response.choices[0].message.content)
```

Happy coding! If you would like to contribute, please read our [Contributing Guide](CONTRIBUTING.md).