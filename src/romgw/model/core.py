from __future__ import annotations

# ----- Warning suppression -----
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN custom operations

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# -------------------------------

import joblib
import keras
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler
from typing import Literal

from romgw.config.env import PROJECT_ROOT
from romgw.waveform.base import FullWaveform, ComponentWaveform
from romgw.waveform.params import PhysicalParams
from romgw.typing.core import (
    RealArray,
    ComplexArray,
    BBHSpinType,
    ModeType,
    ComponentType
)

class ComponentROM:
    """"""

    bbh_spin: BBHSpinType
    mode: ModeType
    component: ComponentType
    name: Literal["NonLinRegV1"]
    param_scaler: ColumnTransformer
    model: keras.Model
    waveform_scaler: MinMaxScaler
    interpolation_matrix: RealArray

    def __init__(
        self,
        bbh_spin: BBHSpinType,
        mode: ModeType,
        component: ComponentType,
        name: Literal["NonLinRegV1"] = "NonLinRegV1",
    ) -> "ComponentROM":
        """"""
        self.bbh_spin = bbh_spin
        self.mode = mode
        self.component = component
        self.name = name

        models_root = PROJECT_ROOT / "models"
        self.param_scaler = joblib.load(models_root / "x_scaler.gz")

        model_dir = models_root / bbh_spin / name / mode / component
        self.model = keras.models.load_model(model_dir / "model.keras")
        self.waveform_scaler = joblib.load(model_dir / "y_scaler.gz")
        self.interpolation_matrix = np.load(model_dir / "interpolation_matrix.npy",
                                            allow_pickle=False)

    def generate_mode_component(self, params_arr: RealArray) -> ComplexArray:
        """"""
        params_scaled = self.param_scaler.transform(params_arr[:, np.newaxis])
        h_nodes_scaled = self.model.predict(params_scaled, verbose=0).astype(np.float64)
        h_nodes = self.waveform_scaler.inverse_transform(h_nodes_scaled).flatten()
        return self.interpolation_matrix @ h_nodes
    
    def generate_mode_component_vectorised(
        self,
        params_arr: RealArray
    ) -> ComplexArray:
        """"""
        params_scaled = self.param_scaler.transform(params_arr)
        h_nodes_scaled = self.model.predict(params_scaled, verbose=0).astype(np.float64)
        h_nodes = self.waveform_scaler.inverse_transform(h_nodes_scaled)
        return (self.interpolation_matrix @ h_nodes.T).T

    def __repr__(self) -> str:
        return (f"<ComponentROM>(name={self.name}, "
                f"bbh_spin={self.bbh_spin}, "
                f"mode={self.mode}, "
                f"component={self.component}"
                f")")


class ModeROM:
    """"""

    bbh_spin: BBHSpinType
    mode: ModeType
    name: Literal["NonLinRegV1"]
    component_models: dict[ComponentType, ComponentROM]

    def __init__(
        self,
        bbh_spin: BBHSpinType,
        mode: ModeType,
        name: Literal["NonLinRegV1"] = "NonLinRegV1",
    ) -> "ModeROM":
        """"""
        self.bbh_spin = bbh_spin
        self.mode = mode
        self.name = name

        self.component_models = {
            "amplitude": ComponentROM(bbh_spin, mode, "amplitude", name),
            "phase": ComponentROM(bbh_spin, mode, "phase", name),
        }

    def generate_mode(
        self,
        params_arr: RealArray,
    ) -> RealArray:
        """"""
        amp_arr = (self.component_models["amplitude"]
                       .generate_mode_component(params_arr))
        phi_arr = (self.component_models["phase"]
                       .generate_mode_component(params_arr))
        return amp_arr * np.exp(-1j * phi_arr)
    
    def generate_mode_vectorised(
        self,
        params_arr: RealArray,
    ) -> RealArray:
        """"""
        amp_arr = (self.component_models["amplitude"]
                       .generate_mode_component_vectorised(params_arr))
        phi_arr = (self.component_models["phase"]
                       .generate_mode_component_vectorised(params_arr))
        return amp_arr * np.exp(-1j * phi_arr)

    def __repr__(self) -> str:
        return (f"<ModeROM>(name={self.name}, "
                f"bbh_spin={self.bbh_spin}, "
                f"mode={self.mode}"
                f")")

    
class ROMGW:
    """"""

    bbh_spin: BBHSpinType
    name: Literal["NonLinRegV1"]
    mode_models: dict[ModeType, ModeROM]

    def __init__(
        self,
        bbh_spin: BBHSpinType,
        name: Literal["NonLinRegV1"] = "NonLinRegV1",
    ) -> "ROMGW":
        """"""
        self.bbh_spin = bbh_spin
        self.name = name

        model_dir = PROJECT_ROOT / "models" / bbh_spin / name
        mode_list = [m.name for m in model_dir.iterdir() if m.is_dir]
        self.mode_models = {m: ModeROM(bbh_spin, m, name)
                            for m in mode_list}

    def generate_modes(
        self,
        params_arr: RealArray
    ) -> dict[ModeType, RealArray]:
        """"""
        return {m: mode_model.generate_mode(params_arr)
                for m, mode_model in self.mode_models.items()}
    
    def generate_modes_vectorised(
        self,
        params_arr: RealArray
    ) -> list[dict[ModeType, RealArray]]:
        """"""
        modes_arr = {m: mode_model.generate_mode_vectorised(params_arr)
                        for m, mode_model in self.mode_models.items()}
        modes = []
        for i in range(len(params_arr)):
            modes.append({mode_key: mode_val[i]
                          for mode_key, mode_val in modes_arr.items()})
        return [
            {mode_key: mode_val[i] for mode_key, mode_val in modes_arr.items()}
            for i in range(len(params_arr))
        ]
    
    def __repr__(self) -> str:
        return (f"<ROMGW>(name={self.name}, "
                f"bbh_spin={self.bbh_spin}, "
                f"modes={list(self.mode_models.values())}"
                f")")
    