import joblib
import numpy as np
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from typing import Literal
from typing_extensions import overload

from romgw.config.env import PROJECT_ROOT
from romgw.typing.utils import validate_literal
from romgw.waveform.dataset import ComponentWaveformDataset
from romgw.typing.core import (
    BBHSpinType,
    ModeType,
    ComponentType,
    RealArray
)

def load_raw_training_data(
    bbh_spin: BBHSpinType,
    mode: ModeType,
    component: ComponentType,
) -> tuple[RealArray, RealArray]:
    """"""
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)
    mode = validate_literal(mode, ModeType)
    component = validate_literal(component, ComponentType)

    data_dir = PROJECT_ROOT / "data" / bbh_spin / "train" / mode / component

    wf_dir = data_dir / "raw"
    waveforms = ComponentWaveformDataset.from_directory(wf_dir,
                                                        component=component)

    empirical_time_nodes_file = (
        data_dir / "empirical_interpolation" / "empirical_time_nodes.npy"
    )
    if not empirical_time_nodes_file.is_file:
        raise FileNotFoundError(f"Could not find the file "
                                f"{empirical_time_nodes_file}")
    empirical_time_nodes = np.load(empirical_time_nodes_file,
                                   allow_pickle=False)

    X_raw = waveforms.params_array
    Y_raw = waveforms.array[:, empirical_time_nodes]

    return (X_raw[:, :1], Y_raw) if bbh_spin == "NS" else (X_raw, Y_raw)


@overload
def save_scaler(scaler: ColumnTransformer, x_or_y: Literal["x"]): ...
@overload
def save_scaler(scaler: MinMaxScaler, x_or_y: Literal["y"]): ...
def save_scaler(
    scaler: ColumnTransformer | MinMaxScaler,
    x_or_y: Literal["x", "y"],
    model_dir: str | Path
) -> None:
    """"""
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    scaler_file = model_dir / f"{x_or_y}_scaler.gz"
    joblib.dump(scaler, scaler_file)


@overload
def load_scaler(x_or_y: Literal["x"]) -> ColumnTransformer: ...
@overload
def load_scaler(x_or_y: Literal["y"]) -> MinMaxScaler: ...
def load_scaler(
    x_or_y: Literal["x", "y"],
    model_dir: str | Path
) -> ColumnTransformer | MinMaxScaler:
    """"""
    validate_literal(x_or_y, Literal["x", "y"])
    scaler_file = model_dir / f"{x_or_y}_scaler.gz"
    if not scaler_file.is_file():
        raise FileNotFoundError(f"Could not find file {scaler_file}")
    scaler = joblib.load(scaler_file)
    return scaler
