import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # disable oneDNN custom operations

import warnings
warnings.filterwarnings('ignore')

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import joblib
import keras
import numpy as np
import typer
from pathlib import Path
from typing import Literal

from romgw.config.env import PROJECT_ROOT
from romgw.typing.core import (
    BBHSpinType,
    ModeType,
)

def working_model_files(
    bbh_spin: BBHSpinType,
    mode: ModeType,
    model_name: Literal["NonLinRegV1"],
) -> dict[str, Path]:
    """"""
    mode_dir = PROJECT_ROOT / "data" / bbh_spin / "train" / mode

    param_scaler = (mode_dir / "amplitude" / "models" / model_name /
                    "x_scaler.gz")
    
    model = lambda c: mode_dir / c / "models" / model_name / "model.keras"
    waveform_scaler = lambda c: (mode_dir / c / "models" / model_name /
                                 "y_scaler.gz")
    interpolation_matrix = lambda c: (mode_dir / c /
                                      "empirical_interpolation" /
                                      "B_matrix.npy")
    
    return {"param_scaler": param_scaler,
            "amp_model": model("amplitude"),
            "amp_scaler": waveform_scaler("amplitude"),
            "amp_interpolation_matrix": interpolation_matrix("amplitude"),
            "phi_model": model("phase"),
            "phi_scaler": waveform_scaler("phase"),
            "phi_interpolation_matrix": interpolation_matrix("phase")}

def final_model_files(
    bbh_spin: BBHSpinType,
    mode: ModeType,
    model_name: Literal["NonLinRegV1"],
) -> dict[str, Path]:
    """"""
    param_scaler = PROJECT_ROOT / "models" / "x_scaler.gz"
    
    model_dir = PROJECT_ROOT / "models" / bbh_spin / model_name / mode

    model = lambda c: model_dir / c / "model.keras"
    waveform_scaler = lambda c: model_dir / c / "y_scaler.gz"
    interpolation_matrix = lambda c: model_dir / c / "interpolation_matrix.npy"

    return {"param_scaler": param_scaler,
            "amp_model": model("amplitude"),
            "amp_scaler": waveform_scaler("amplitude"),
            "amp_interpolation_matrix": interpolation_matrix("amplitude"),
            "phi_model": model("phase"),
            "phi_scaler": waveform_scaler("phase"),
            "phi_interpolation_matrix": interpolation_matrix("phase")}

def finalise_model_files(
    working_model_files: dict[str, Path],
    final_model_files: dict[str, Path],
) -> None:
    """"""
    for file in final_model_files.values():
        file.parent.mkdir(parents=True, exist_ok=True)

    # param scaler
    param_scaler = joblib.load(working_model_files["param_scaler"])
    joblib.dump(param_scaler, final_model_files["param_scaler"])

    # models
    amp_model = keras.models.load_model(working_model_files["amp_model"])
    amp_model.save(final_model_files["amp_model"])
 
    phi_model = keras.models.load_model(working_model_files["phi_model"])
    phi_model.save(final_model_files["phi_model"])

    # waveform scalers
    amp_scaler = joblib.load(working_model_files["amp_scaler"])
    joblib.dump(amp_scaler, final_model_files["amp_scaler"])

    phi_scaler = joblib.load(working_model_files["phi_scaler"])
    joblib.dump(phi_scaler, final_model_files["phi_scaler"])

    # interpolation matrices
    amp_interpolation_matrix = np.load(
        file=working_model_files["amp_interpolation_matrix"],
        allow_pickle=False
    )
    np.save(final_model_files["amp_interpolation_matrix"],
            amp_interpolation_matrix,
            allow_pickle=False)

    phi_interpolation_matrix = np.load(
        file=working_model_files["phi_interpolation_matrix"],
        allow_pickle=False
    )
    np.save(final_model_files["phi_interpolation_matrix"],
            phi_interpolation_matrix,
            allow_pickle=False)

HELP_BBH_SPIN = 'Spin configuration: "NS" = no-spin, "AS" = aligned-spin, "PS" = precessing-spin.'
HELP_MODE = 'Spin-weighted spherical harmonic label (l, m): "2,2", "2,1", "3,2", "3,3", "4,4", or "4,3".'
HELP_MODEL_NAME = 'Name of the model: "NonLinRegV1"'
HELP_VERBOSE = "Enable verbose output."

app = typer.Typer(help="Reduced Order Modelling of Gravitational Waves CLI")

@app.command()
def main(
    bbh_spin: BBHSpinType = typer.Option(..., help=HELP_BBH_SPIN),
    mode: ModeType = typer.Option(..., help=HELP_MODE),
    model_name: Literal["NonLinRegV1"] = typer.Option(..., help=HELP_MODEL_NAME),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help=HELP_VERBOSE),
) -> None:
    """
    Examples
    --------
    Finalise the NonLinRegV1 reduced order model for the 2,2 mode
    of the GW produced by the merger of an NS BBH:

        $ finalise_mode_rom.py --bbh-spin NS --mode 2,2 --model-name NonLinRegV1
    """
    working_files = working_model_files(bbh_spin, mode, model_name)
    if verbose:
        typer.echo("working files:")
        for object, file in working_files.items():
            typer.echo(f"  {object}:\n    {file}")
        typer.echo("")

    final_files = final_model_files(bbh_spin, mode, model_name)
    if verbose:
        typer.echo("finalised files:")
        for object, file in final_files.items():
            typer.echo(f"  {object}:\n    {file}")
    
    finalise_model_files(working_files, final_files)

if __name__ == "__main__":
    app()
