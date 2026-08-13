from __future__ import annotations


class ImageGeneratorError(Exception):
    """Base error with a safe message that may be shown in chat."""

    def __init__(self, public_message: str, *, detail: str = "") -> None:
        super().__init__(detail or public_message)
        self.public_message = public_message
        self.detail = detail or public_message


class ConfigurationError(ImageGeneratorError):
    """Raised when a required plugin setting is missing or invalid."""


class ImageAPIError(ImageGeneratorError):
    """Raised when the upstream image service cannot complete a request."""

