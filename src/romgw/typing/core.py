from numpy.typing import NDArray
from typing import Literal
from typing_extensions import Annotated, TypeAlias
import numpy as np


# ----------------------------------------------
# TYPE ALIASES
# ----------------------------------------------
RealArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]

BBHSpinType: TypeAlias = Literal["NS", "AS", "PS"]
DatasetType: TypeAlias = Literal["train", "test"]
ModeType: TypeAlias = Literal["2,2", "2,1", "3,3", "3,2", "4,4", "4,3"]
ComponentType: TypeAlias = Literal["amplitude", "phase"]

MassRatio: TypeAlias = float
SpinScalar: TypeAlias = float
SpinVector: TypeAlias = Annotated[NDArray[np.float64], "(3,)"]
Spin: TypeAlias = SpinScalar | SpinVector


# ----------------------------------------------
# RUNTIME VALUE SETS
# ----------------------------------------------
BBHSPIN_VALUES: set[str] = {"NS", "AS", "PS"}
DATASET_VALUES: set[str] = {"train", "test"}
MODE_VALUES: set[str] = {"2,2", "2,1", "3,3", "3,2", "4,4", "4,3"}
COMPONENT_VALUES: set[str] = {"amplitude", "phase"}
