from easyai4all.providers.options.anthropic import Anthropic


provider = Anthropic()

response = provider.create(
    model="claude-3-5-sonnet-20240620",
    messages=[{"role": "user", "content": "What's the weather in SF?"}],
)

print(response)
