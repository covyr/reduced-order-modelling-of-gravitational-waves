from typing import Any
import numpy as np

from romgw.typing.exceptions import InvalidLiteralValueError
from romgw.typing.helpers import literal_values
from romgw.typing.core import (
    MassRatio,
    Spin,
)

# ----------------------------------------------
# RUNTIME VALIDATION
# ----------------------------------------------
def validate_literal(value: str, literal_type: Any) -> str:
    """Validate that a string is one of the allowed Literal values."""
    allowed = literal_values(literal_type)
    if value not in allowed:
        raise InvalidLiteralValueError(value, literal_type)
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

from romgw.typing.core import (
    BBHSPIN_VALUES,
    MODE_VALUES,
    COMPONENT_VALUES,
    DATASET_VALUES
)

def validate_dataset(dataset):
    """
    Raise ValueError if not a valid DatasetType
    """
    return


def validate_bbh_spin(bbh_spin):
    """
    Raise ValueError if not a valid BBHSpinType.
    """
    if bbh_spin not in BBHSPIN_VALUES:
        raise ValueError(f"BBH spin {bbh_spin} not valid")


def validate_mode(bbh_spin, mode):
    """
    Raise ValueError if not a valid ModeType for the given BBHSpinType.
    """
    if mode not in MODE_VALUES[bbh_spin]:
        raise ValueError(f"Mode {mode} not valid for bbh_spin {bbh_spin}")


def validate_component(component):
    """
    Raise ValueError if not a valid ComponentType.
    """
    if component not in COMPONENT_VALUES:
        raise ValueError(f"Component {component} not valid")
