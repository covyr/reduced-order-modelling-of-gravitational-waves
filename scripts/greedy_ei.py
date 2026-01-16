import numpy as np
import typer

from romgw.config.constants import PROJECT_ROOT, MODE_VALUES
from romgw.maths.ei import empirical_time_nodes
# from romgw.typing.core import BBHSpinType, ModeType, ComponentType
from romgw.config.types import (
    BBHSpinType,
    DatasetType,
    ModeType,
    ComponentType
)
# from romgw.typing.utils import validate_literal
from romgw.config.validation import (
    validate_literal,
    validate_dependent_literal
)
from romgw.waveform.dataset import ComponentWaveformDataset

HELP_BBH_SPIN = 'Spin configuration: "NS" = no-spin, "AS" = aligned-spin, "PS" = precessing-spin.'
HELP_DATASET = 'Dataset label: "train_xl" = xl training data, "train" = training data, "test" = testing data.'
HELP_MODE = 'Spin-weighted spherical harmonic label (l, m): "2,2", "2,1", "3,2", "3,3", "4,4", or "4,3".'
HELP_COMPONENT = 'Waveform component: "amplitude" or "phase".'
HELP_SAVING = "Whether to save empirical time nodes and B matrix."
HELP_VERBOSE = "Enable verbose output."

app = typer.Typer(help="Reduced Order Modelling of Gravitational Waves CLI")

@app.command()
def main(
    bbh_spin: BBHSpinType = typer.Option(..., help=HELP_BBH_SPIN),
    dataset: DatasetType = typer.Option(..., help=HELP_DATASET),
    mode: ModeType = typer.Option(..., help=HELP_MODE),
    component: ComponentType = typer.Option(..., help=HELP_COMPONENT),
    saving: bool = typer.Option(False, "--saving/--no-saving", help=HELP_SAVING),
    verbose: bool = typer.Option(False, "--verbose/--no-verbose", help=HELP_VERBOSE),
) -> None:
    """
    Find empirical time nodes for waveforms using a greedy algorithm.

    Employs a greedy algorithm to select the time nodes at which the training
    waveforms demonstrate the greatest variation. This criterion for selection
    ensures that each node selected (of which there are m in total) is the
    most informative node, which maximally reduces the mismatch between a
    fiducial waveform and its corresponding empirical interpolant, constructed
    using the B matrix which is derived from the empirical time nodes.

    Parameters
    ----------
    bbh_spin : BBHSpinType
        Spin configuration of the BBH. Can be "NS" (no-spin),
        "AS" (aligned-spin), or "PS" (precessing-spin).
    mode : ModeType
        Spin-weighted spherical harmonic mode label (e.g. "2,2", "3,3").
    component : ComponentType
        Waveform component (full modes are decomposed into their amplitude
        and phase parts). Can be "amplitude" or "phase".
    saving : bool, optional
        If True, saves the empirical time nodes and B matrix returned by
        `empirical_time_nodes` calls. Default is False.
    verbose : bool, optional
        If True, prints progress updates. Default is False.

    Returns
    -------
    None

    See Also
    --------
    empirical_time_nodes : Greedy algorithm implementation for empirical time
                           node selection.

    Examples
    --------
    Find and save the empirical time nodes and B matrix for the amplitude of
    the 2,2 mode of the GW produced by the merger of an NS BBH for the
    training set:

        $ python greedy_ei.py --bbh-spin NS --dataset train --mode 2,2 --component amplitude --saving
    """
    bbh_spin = validate_literal(
        value=bbh_spin,
        literal_type=BBHSpinType,
    )
    dataset = validate_literal(
        value=dataset,
        literal_type=DatasetType,
    )
    mode = validate_dependent_literal(
        value=mode,
        literal_type=ModeType,
        parent_value=bbh_spin,
        parent_literal_type=BBHSpinType,
        dependency_map=MODE_VALUES,
    )
    component = validate_literal(
        value=component,
        literal_type=ComponentType,
    )

    if verbose:
        typer.echo(f"Finding empirical times for "
                   f"{bbh_spin=}, {mode=}, {component=}")
    
    data_dir = PROJECT_ROOT / "data" / bbh_spin / dataset / mode / component

    rb_dir = data_dir / "reduced_basis" / "elements"
    rb = ComponentWaveformDataset.from_directory(rb_dir, component=component)
    
    etns, B = empirical_time_nodes(rb)
    if verbose:
        typer.echo("Greedy empirical time node selection complete.")

    if len(etns) != len(set(etns)):  # empirical time nodes must be unique
        raise ValueError("Empirical time nodes must be unique.")
    
    if saving:
        ei_dir = data_dir / "empirical_interpolation"
        ei_dir.mkdir(parents=True, exist_ok=True)
        
        empirical_time_node_file = ei_dir / "empirical_time_nodes.npy"
        np.save(empirical_time_node_file, etns, allow_pickle=False)

        B_matrix_file = ei_dir / "B_matrix.npy"
        np.save(B_matrix_file, B, allow_pickle=False)

if __name__ == "__main__":
    app()
