from typing import Literal
from typing_extensions import Annotated, TypeAlias
import numpy as np
import numpy.typing as npt


# ----------------------------------------------
# TYPE ALIASES
# ----------------------------------------------
RealArray: TypeAlias = npt.NDArray[np.float32 | np.float64]
ComplexArray: TypeAlias = npt.NDArray[np.complex64 | np.complex128]

DatasetType: TypeAlias = Literal["test", "train", "train_xl"]
ModelNameType: TypeAlias = Literal["NonLinRegV1"]

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
