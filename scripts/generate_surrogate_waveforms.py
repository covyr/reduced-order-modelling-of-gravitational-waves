from scipy.interpolate import InterpolatedUnivariateSpline
from typing import Dict, Tuple
import numpy as np
import pyseobnr
import time

from romgw.config.env import COMMON_TIME, PROJECT_ROOT
from romgw.generate_waveform import generate_modes
from romgw.typing.utils import validate_literal
from romgw.utils.filesystem import empty_directory
from romgw.waveform.params import PhysicalParams
from romgw.waveform.stat import ModelStat
from romgw.waveform.base import (
    ComponentWaveform,
    FullWaveform,
)
from romgw.typing.core import (
    RealArray,
    ComplexArray,
    MassRatio,
    SpinScalar,
    SpinVector,
    BBHSpinType,
    DatasetType,
    MODE_VALUES,
)


def generate_romgw_waveform(
    params: PhysicalParams,
):
    precessing = False if np.ndim(params.chi1) == 0 else True

    # Retrieve mass ratio from PhysicalParams.
    q: MassRatio = params.q

    # Retrieve spins from PhysicalParams.
    if precessing:  # RealArray of shape (3,)
        chi1: SpinVector = params.chi1
        chi2: SpinVector = params.chi2
    else:  # float
        chi1: SpinScalar = params.chi1
        chi2: SpinScalar = params.chi2

    # Generate the waveform modes and time it
    start = time.perf_counter()
    modes = generate_modes(

    )


def main(
    bbh_spin: BBHSpinType,
    dataset: DatasetType,
    saving: bool = False
) -> None:
    """
    Generate fiducial waveforms for a dataset using SEOBNRv5.

    Generates waveforms for a spin-specific dataset using the parameter space
    created by `generate_parameter_space.py`. For each waveform, saves the
    complex waveform, its amplitude, phase, and generation statistics.

    Parameters
    ----------
    spin : {"NS", "AS", "PS"}
        The complexity regarding the BBH spins. "NS" for no-spin, "AS" for
        aligned-spin, or "P" for precessing.
    dataset : {"train", "test"}
        The dataset type. "train" for training data (larger) or "test" for
        testing data (smaller).
    saving: bool
        Whether to save/perform filesystem operations or not.

    Returns
    -------
    None

    See Also
    --------
    generate_seobnrv5_waveform : Generate waveform with SEOBNRv5.
    """
    # Validate literals. Raises exception if invalid.
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)
    dataset = validate_literal(dataset, DatasetType)

    # Every directory in this script is a child of this one.
    dataset_dir = PROJECT_ROOT / "data" / bbh_spin / dataset

    # Load the parameter space.
    param_space_filepath = dataset_dir / "parameter_space.npy"
    param_space = np.load(param_space_filepath)

    # For filename/printing purposes.
    lsM = len(str((M := param_space.shape[0]) - 1))

    # Empty directories to save to,
    # to avoid issues with reusing directories.
    if saving:
        # Generation time directory.
        stat_dir = dataset_dir / "stats"
        stat_dir.mkdir(parents=True, exist_ok=True)
        empty_directory(stat_dir)

        # Mode directories.
        for path in dataset_dir.iterdir():
            if path.is_dir() and path.name in MODE_VALUES:
                for component in ("full", "amplitude", "phase"):
                    wf_dir = path / component / "raw"
                    wf_dir.mkdir(parents=True, exist_ok=True)
                    empty_directory(wf_dir)

    for i, params in enumerate(param_space):
        print(f"Generating waveform {i+1:0{lsM}d}/{M}", end='\r')

        # Retrieve mass ratio from params array.
        q: MassRatio = float(params[0])

        # Retrieve spins from params array.
        if bbh_spin == "PS":
            chi1: SpinVector = params[1:4].astype(np.float64)
            chi2: SpinVector = params[4:7].astype(np.float64)
        else:
            chi1: SpinScalar = float(params[1])
            chi2: SpinScalar = float(params[2])
        
        # Instantiate params as PhysicalParams,
        # ensuring enforcement of param constraints.
        physical_params = PhysicalParams(q, chi1, chi2)
        
        # Generate waveform with fiducial model.
        modes, stat = generate_romgw_waveform(params=physical_params)
        # print(f"{modes=}")
        # print(f"{stat=}")

        # Save stat.
        if saving:
            stat_file = stat_dir / f"stat_{i:0{lsM}d}.json"
            stat.to_file(stat_file)

        for mode in modes:
            # Full mode waveform.
            waveform: FullWaveform = modes[mode]
            # Amplitude for mode waveform.
            amplitude: ComponentWaveform = waveform.amplitude
            # Phase for mode waveform.
            phase: ComponentWaveform = waveform.phase

            # Saving.
            if saving:
                # Set filename for waveforms to save to.
                filename = f"waveform_{i:0{lsM}d}.npz"

                # 'Mode-anchored' directory.
                mode_dir = dataset_dir / mode

                # Full mode waveform.
                wf_full_file = mode_dir / "full" / "raw" / filename
                waveform.to_file(wf_full_file)

                # Amplitude of mode waveform.
                wf_amplitude_file = mode_dir / "amplitude" / "raw" / filename
                amplitude.to_file(wf_amplitude_file)

                # Phase of mode waveform.
                wf_phase_file = mode_dir / "phase" / "raw" / filename
                phase.to_file(wf_phase_file)
                
    print("Waveform generation complete.")


if __name__ == "__main__":
    main(bbh_spin="NS",
         dataset="test",
         saving=True)
