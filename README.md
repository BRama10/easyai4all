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

> [!TIP] 
>
> Unlike `aisuite` and `litellm`, we directly interact with the LLMs via REST API's over HTTPS, meaning no external client dependencies or abstractions. This allows `easyai4all` to be extremely lightweight (only one dependency - `httpx`)!


## Installation

You can install `easyai4all` via PyPI

```shell
pip install easyai4all
```

## Set up

<!-- To get started, you will need API Keys for the providers you intend to use. You'll need to
install the provider-specific library either separately or when installing aisuite.

The API Keys can be set as environment variables, or can be passed as config to the aisuite Client constructor.
You can use tools like [`python-dotenv`](https://pypi.org/project/python-dotenv/) or [`direnv`](https://direnv.net/) to set the environment variables manually. Please take a look at the `examples` folder to see usage.

Here is a short example of using `aisuite` to generate chat completion responses from gpt-4o and claude-3-5-sonnet.

Set the API keys.
```shell
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

Use the python client.
```python
import aisuite as ai
client = ai.Client()

models = ["openai:gpt-4o", "anthropic:claude-3-5-sonnet-20240620"]

messages = [
    {"role": "system", "content": "Respond in Pirate English."},
    {"role": "user", "content": "Tell me a joke."},
]

for model in models:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.75
    )
    print(response.choices[0].message.content)

```
Note that the model name in the create() call uses the format - `<provider>:<model-name>`.
`aisuite` will call the appropriate provider with the right parameters based on the provider value.
For a list of provider values, you can look at the directory - `aisuite/providers/`. The list of supported providers are of the format - `<provider>_provider.py` in that directory. We welcome  providers adding support to this library by adding an implementation file in this directory. Please see section below for how to contribute.

For more examples, check out the `examples` directory where you will find several notebooks that you can run to experiment with the interface. -->

## License

`easyai4all` is released under the MIT License. You are free to use, modify, and distribute the code for both commercial and non-commercial purposes.

## Contributing

<!-- If you would like to contribute, please read our [Contributing Guide](https://github.com/andrewyng/aisuite/blob/main/CONTRIBUTING.md) and join our [Discord](https://discord.gg/T6Nvn8ExSb) server! -->

## Adding support for a provider
We have made easy for a provider or volunteer to add support for a new platform.

TBD