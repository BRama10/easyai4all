# easyai4all

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Truly unified & comprehensive interface to multiple Generative AI providers.

Inspired from `aisuite` by Andrew Ng and `litellm` by BerriAI, `easyai4all` unifies the various LLM providers under a single interface. Derived from the OpenAI specification, `easyai4all` allows users to interact with all kinds of LLMs, with a standardized input/output format. `easyai4all` is a comprehensive wrapper, meaning that all functionalities supported by the individual LLM providers are available through `easyai4all`.

Currently supported providers, along with functionalities are -

| LLM Provider | Is Supported | JSON Mode | Tool Calling |
|--------------|--------------|-----------|--------------|
| OpenAI | ✅ | ✅ | ✅ |
| Anthropic | ✅ | ✅ | ❌ |
| Google (Gemini) | ✅ | ✅ | ✅ |

>
>
> Unlike `aisuite` and `litellm`, we directly interact with the LLMs via REST API's over HTTPS, meaning no external client dependencies or abstractions. This allows `easyai4all` to be extremely lightweight (only one dependency - `httpx`) and doesn't require you to install **any** client libraries!


## Installation

You can install `easyai4all` via PyPI

```shell
pip install easyai4all
```

## Set up

To get started, you will need API Keys for the providers you intend to use. That's all!

The API Keys can be set as environment variables, or can be passed as config to the easyai4all Client constructor. We handle loading `.env` files so you don't need to do anything extra!


Here is a short example of using `easyai4all` to generate chat completion responses from gpt-4o and claude-3-5-sonnet.

Set the API keys.
```shell
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Use the python client.
```python
from easyai4all.client import Client

client = Client()

models = ["openai/gpt-4o", "anthropic/claude-3-5-sonnet-20240620"]

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

for model in models:
    response = clientcreate(
        model=model,
        messages=messages,
        temperature=0.75
    )

    print(response.choices[0].message.content)

```

Note that the model name in the create() call uses the format - `<provider>/<model-name>`.
`easyai4all` will call the appropriate provider with the right parameters based on the provider value.

For a list of provider values, you can check the table above. We welcome providers adding support to this library by adding an implementation file in this directory. Please see section below for how to contribute.

For more examples, check out the `examples` directory where you will find several notebooks that you can run to experiment with the interface.

## License

`easyai4all` is released under the MIT License. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.

## Contributing

If you would like to contribute, please read our [Contributing Guide](https://github.com/BRama10/easyai4all/blob/main/CONTRIBUTING.md)!

## Adding support for a provider
We have made easy for a provider or volunteer to add support for a new platform.

In the `easyai4all/providers/options` folder, add a provider file in the format `{providername in lowercase}.py`. Inherit the `Provider` class (`from easyai4all.providers.base_provider import Provider`).

Implement the `_prepare_request` and `_prepare_response` methods, where `_prepare_request` should return the request data in a format compatible with the provider's REST API and `_process_response` should return a `ChatCompletionResponse` object containing the provider's response properly formatted.

Example implementation below ->

```python
def _prepare_request(
    self, model: str, messages: List[Dict[str, Any]], **kwargs
) -> Dict[str, Any]:
    return {"model": model, "messages": messages, **kwargs}

def _process_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
    return ChatCompletionResponse.from_dict(response)
```
