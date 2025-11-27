import numpy as np

from romgw.config.env import PROJECT_ROOT
from romgw.maths.ei import empirical_time_nodes
from romgw.typing.core import BBHSpinType, ModeType, ComponentType
from romgw.typing.utils import validate_literal
from romgw.waveform.dataset import ComponentWaveformDataset

def main(
    bbh_spin: BBHSpinType,
    mode: ModeType,
    component: ComponentType,
    saving: bool = False,
    verbose: bool = False,
) -> None:
    """"""
    # Validate literals. Raises exception if invalid.
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)
    mode = validate_literal(mode, ModeType)
    component = validate_literal(component, ComponentType)

    if verbose:
        print(f"Finding empirical times for {bbh_spin=}, {mode=}, {component=}")
    
    data_dir = PROJECT_ROOT / "data" / bbh_spin / "train" / mode / component

    rb_dir = data_dir / "reduced_basis" / "elements"
    rb = ComponentWaveformDataset.from_directory(rb_dir, component=component)
    
    etns, B = empirical_time_nodes(rb)
    if verbose:
        print(f"{etns=}\n{etns.shape=}")
        print(f"{B=}\n{B.shape=}")

    if len(etns) != len(set(etns)):  # empirical time nodes must be unique
        raise ValueError("Empirical time nodes must be unique.")
    
    if saving:
        ei_dir = data_dir / "empirical_interpolation"
        
        empirical_time_node_file = ei_dir / "empirical_time_nodes.npy"
        np.save(empirical_time_node_file, etns, allow_pickle=False)

        B_matrix_file = ei_dir / "B_matrix.npy"
        np.save(B_matrix_file, B, allow_pickle=False)


if __name__ == "__main__":
    main(bbh_spin="NS",
         mode="2,2",
         component="amplitude",
         saving=True,
         verbose=True)
