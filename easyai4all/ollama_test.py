from providers.options.ollama import Ollama

def run_ollama_chat():
    # Initialize the Ollama provider
    ollama = Ollama(api_base="http://localhost:11434/api/chat")

    # Define the chat completion parameters
    model = "llama3.2"
    messages = [
        {"role": "user", "content": "What is the weather in Paris?"},
    ]
    # Optionally, define tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The location to get the weather for, e.g. Paris, France"
                        },
                        "format": {
                            "type": "string",
                            "description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
                            "enum": ["celsius", "fahrenheit"]
                        }
                    },
                    "required": ["location", "format"]
                }
            }
        }
    ]
    stream = False

    # Send the chat request
    try:
        response = ollama.create(model=model, messages=messages, tools=tools, stream=stream)

        # Print the processed response
        print(response.to_dict())
    except Exception as e:
        print(f"Error during Ollama chat completion: {e}")

if __name__ == "__main__":
    run_ollama_chat()

