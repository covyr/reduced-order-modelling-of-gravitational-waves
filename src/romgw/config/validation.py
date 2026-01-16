from typing import Any
import numpy as np

from .types import (
    MassRatio,
    Spin,
)


# -----------------------------------------------------------------------------
# BASE EXCEPTION
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# EXCEPTIONS
# -----------------------------------------------------------------------------
class InvalidLiteralValueError(ROMGWError):
    """Raised when a string value doesn't match a defined Literal type."""

    default_message = (
        "Invalid value '{value}' for {literal_name}. "
        "Allowed values: {allowed}."
    )

    def __init__(
        self,
        value: str,
        literal_type: Any,
        allowed: set[str] | None = None,
    ):
        if allowed is None:
            from typing import get_args
            try:
                allowed = set(get_args(literal_type))
            except Exception:
                allowed = []
        literal_name = getattr(literal_type, "__name__", str(literal_type))
        message = self.default_message.format(
            value=value,
            literal_name=literal_name,
            allowed=allowed,
        )
        super().__init__(message, value=value, literal_type=literal_type)


# -----------------------------------------------------------------------------
# RUNTIME VALIDATION
# -----------------------------------------------------------------------------
def validate_literal(
    value: str,
    literal_type: Any,
    allowed: set[str] | None = None,
) -> str:
    """Validate that a string is one of the allowed Literal values."""
    from typing import get_args
    if allowed is None:
        from typing import get_args
        try:
            allowed = set(get_args(literal_type))
        except Exception:
            allowed = []
    if value not in allowed:
        raise InvalidLiteralValueError(value, literal_type, allowed)
    return value


def validate_dependent_literal(
    value: str,
    literal_type: Any,
    parent_value: str,
    parent_literal_type: Any,
    dependency_map: dict[str, set[str]]
) -> str:
    """
    Validate a value that depends on a parent value.

    Example:
        mode = validate_dependent_literal("2,2", ModeType, "NS", BBHSpinType, MODE_VALUES)
    """
    allowed = dependency_map.get(parent_value, None)
    if allowed is None:
        raise InvalidLiteralValueError(parent_value, parent_literal_type)
    if value not in allowed:
        raise InvalidLiteralValueError(value, literal_type, allowed)
    return value


def validate_mass_ratio(x) -> MassRatio:
    """
    Ensure that ``x`` is a valid mass ratio (q).

    Parameters
    ----------
    x : array_like or float
        Value to validate.

    Returns
    -------
    q : float
        Validated mass ratio in the range [1, ∞).

    Raises
    ------
    ValueError
        If ``x`` is not scalar or is less than 1.
    """
    if np.ndim(x) != 0:
        arr = np.asarray(x)
        raise ValueError(f"Mass ratio must be scalar, got {arr.shape}.")

    x = float(x)

    if x < 1:
        raise ValueError(f"Mass ratio must be in [1, inf), got {x}.")

    return x


def validate_spin(x) -> Spin:
    """
    Ensure that ``x`` is a valid spin scalar or vector.

    Parameters
    ----------
    x : array_like or float
        Spin value(s) to validate. May be scalar in [-1, 1] or
        a 3D vector with components in [-1, 1].

    Returns
    -------
    spin : float or ndarray of shape (3,)
        Validated spin representation.

    Raises
    ------
    ValueError
        If ``x`` is outside [-1, 1] or has invalid shape.
    """
    if np.ndim(x) == 0:
        x = float(x)
        if not -1 <= x <= 1:
            raise ValueError(f"Spin must be in [-1, 1], got {x}.")
        return x

    arr = np.asarray(x, dtype=np.float64)

    if arr.shape != (3,):
        raise ValueError(f"Spin vector must have shape (3,), "
                         f"got {arr.shape}.")

    if not np.all((-1 <= arr) & (arr <= 1)):
        raise ValueError(f"Spin vector components must be in [-1, 1], "
                         f"got {arr}.")

    return arr
