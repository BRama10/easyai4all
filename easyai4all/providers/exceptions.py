class ProviderError(Exception):
    """Custom exception for LLM errors."""

    def __init__(self, message):
        super().__init__(message)
