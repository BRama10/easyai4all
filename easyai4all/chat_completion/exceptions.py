# chat_completion/exceptions.py
class ChatCompletionError(Exception):
    """Base exception for chat completion errors."""

    pass


class InvalidDataError(ChatCompletionError):
    """Raised when invalid data is provided."""

    pass


class ValidationError(ChatCompletionError):
    """Raised when data validation fails."""

    pass
