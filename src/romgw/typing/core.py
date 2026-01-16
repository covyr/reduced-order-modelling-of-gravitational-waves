from typing import Literal, Any
from typing_extensions import Annotated, TypeAlias
import numpy as np
import numpy.typing as npt


# ----------------------------------------------
# TYPE ALIASES
# ----------------------------------------------
RealArray: TypeAlias = npt.NDArray[np.float32 | np.float64]
ComplexArray: TypeAlias = npt.NDArray[np.complex64 | np.complex128]

DatasetType: TypeAlias = Literal["test", "train", "train_xl"]

BBHSpinType: TypeAlias = Literal["NS", "AS", "PS"]
ModeType: TypeAlias = Literal[
    "2,2", "2,1", "2,0", "2,-1", "2,-2",
    "3,3", "3,2", "3,1", "3,0", "3,-1", "3,-2", "3,-3",
    "4,4", "4,3", "4,2", "4,1", "4,0", "4,-1", "4,-2", "4,-3", "4,-4"
]
ComponentType: TypeAlias = Literal["amplitude", "phase"]

MassRatio: TypeAlias = float
SpinScalar: TypeAlias = float
SpinVector: TypeAlias = Annotated[npt.NDArray[np.float32 | np.float64], "(3,)"]
Spin: TypeAlias = SpinScalar | SpinVector


# ----------------------------------------------
# RUNTIME VALUE SETS
# ----------------------------------------------
BBHSPIN_VALUES: set[str] = {"NS", "AS", "PS"}
DATASET_VALUES: set[str] = {"test", "train", "train_xl"}
MODE_VALUES: dict[str, set[str]] = {
    "NS": {"2,2", "2,1", "3,3", "3,2", "4,4", "4,3"},
    "AS": {"2,2", "2,1", "3,3", "3,2", "4,4", "4,3"},
    "PS": {
        "2,2", "2,1", "2,0", "2,-1", "2,-2",
        "3,3", "3,2", "3,1", "3,0", "3,-1", "3,-2", "3,-3",
        "4,4", "4,3", "4,2", "4,1", "4,0", "4,-1", "4,-2", "4,-3", "4,-4",
    },
}
COMPONENT_VALUES: set[str] = {"amplitude", "phase"}


# ----------------------------------------------
# EXCEPTIONS
# ----------------------------------------------
class InvalidLiteralValueError(ValueError):
    """Raised when a value is not valid for a given Literal or dependent type."""
    def __init__(self, value: Any, expected_type: Any):
        super().__init__(f"Invalid value {value!r} for {expected_type}")


# ----------------------------------------------
# VALIDATION HELPERS
# ----------------------------------------------
def validate_literal(value: str, literal_type: Any) -> str:
    """Validate a string against a standard Literal type."""
    from typing import get_args
    # allowed = set(get_args(literal_type))
    try:
        allowed = set(get_args(literal_type))
    except Exception:
        allowed = []
    if value not in allowed:
        raise InvalidLiteralValueError(value, literal_type)
    return value


def validate_dependent_literal(
    value: str,
    parent_value: str,
    dependency_map: dict[str, set[str]]
) -> str:
    """
    Validate a value that depends on a parent value.

    Example:
        mode = validate_dependent_literal("2,2", "NS", MODE_VALUES)
    """
    allowed = dependency_map.get(parent_value)
    if allowed is None:
        raise InvalidLiteralValueError(parent_value, "Parent type")
    if value not in allowed:
        raise InvalidLiteralValueError(value, f"Allowed values for parent {parent_value}")
    return value


def validate_mode_for_spin(mode: str, spin: BBHSpinType) -> str:
    """Convenience wrapper to validate ModeType against a BBHSpinType."""
    return validate_dependent_literal(mode, spin, MODE_VALUES)
