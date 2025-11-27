from typing import Any

from romgw.typing.core import SpinScalar, SpinVector, Spin
from romgw.typing.helpers import literal_values


class ROMGWError(Exception):
    """Base exception for all ROMGW-related errors with formatting support."""

    default_message: str = "An unspecified ROMGW error occurred."

    def __init__(self, *args, **kwargs):
        if args:
            message = args[0]
        else:
            try:
                message = self.default_message.format(**kwargs)
            except KeyError:
                message = self.default_message
        super().__init__(message)
        self.context = kwargs  # store extra info for debugging
    

class InvalidLiteralValueError(ROMGWError):
    """Raised when a string value doesn't match a defined Literal type."""

    default_message = (
        "Invalid value '{value}' for {literal_name}. "
        "Allowed values: {allowed}."
    )

    def __init__(self, value: str, literal_type: Any):
        allowed = literal_values(literal_type)
        literal_name = getattr(literal_type, "__name__", str(literal_type))
        message = self.default_message.format(
            value=value,
            literal_name=literal_name,
            allowed=allowed,
        )
        super().__init__(message, value=value, literal_type=literal_type)
        