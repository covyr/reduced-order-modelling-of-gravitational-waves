from numpy.typing import NDArray
import numpy as np
import typer

from romgw.config.constants import PROJECT_ROOT
# from romgw.typing.utils import validate_literal
# from romgw.typing.core import RealArray, BBHSpinType, DatasetType
from romgw.config.types import RealArray, BBHSpinType, DatasetType
from romgw.config.validation import validate_literal

# Sizes of training and test datasets.
N = {
    "train_xl": 65536,  # 2**16
    "train": 4096,  # 2**12
    "test": 256,  # 2**8
}

# Mass ratio domain over which to sample.
MASS_RATIO_DOMAIN = (1, 10)

# Spherical polar spin component domains over which to sample.
SPIN_MAG_DOMAIN = (-1.0, 1.0)
SPIN_THETA_DOMAIN = (-np.pi/2, np.pi/2)
SPIN_PHI_DOMAIN = (0.0, np.pi)

# Reproducibility.
SEEDS = {
    "train_xl": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "train": [19, 29, 59, 89, 229, 521, 599, 1129, 1229, 1259, 2591],
    "test": [2, 73, 179, 283, 419, 547, 661, 811, 947, 1087, 1229],
}

def uniform_random_space(
    n: int,
    low: float,
    high: float,
    seed: int | None = None
) -> NDArray[np.float64]:
    """"""
    dist_size = n * 16
    distribution = np.random.uniform(low, high, size=dist_size)
    rng = np.random.default_rng(seed)
    random_idxs = rng.permutation(dist_size)[:n]
    space = distribution[random_idxs].reshape(n, 1).astype(np.float64)
    return space


def spherpol_to_cartesian(
    spherical_polars: RealArray
) -> RealArray:
    """"""
    mag, theta, phi = np.split(spherical_polars, 3, axis=1)
    x = mag * np.sin(theta) * np.cos(phi)
    y = mag * np.sin(theta) * np.sin(phi)
    z = mag * np.cos(theta)
    cartesians = np.hstack([x, y, z]).astype(np.float64)  # shape (n, 3)
    return cartesians


def generate_parameter_space(
    bbh_spin: BBHSpinType,
    n: int,
    seeds: list[int] | None = None
) -> NDArray[np.float64]:
    """"""
    # Validate literals. Raises exception if invalid.
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)

    if not seeds:
        seeds = [None] * 11
    
    if bbh_spin == "PS":
        q_space = uniform_random_space(n, *MASS_RATIO_DOMAIN, seed=seeds[0])
        
        chi1_magnitudes = uniform_random_space(n,
                                               *SPIN_MAG_DOMAIN,
                                               seed=seeds[1])
        chi1_thetas = uniform_random_space(n,
                                           *SPIN_THETA_DOMAIN,
                                           seed=seeds[2])
        chi1_phis = uniform_random_space(n,
                                         *SPIN_PHI_DOMAIN,
                                         seed=seeds[3])
        chi1_spherical_polars = (
            np.hstack([chi1_magnitudes, chi1_thetas, chi1_phis])
            .astype(np.float64)
        )
        chi1_space = spherpol_to_cartesian(chi1_spherical_polars)

        chi2_magnitudes = uniform_random_space(n,
                                               *SPIN_MAG_DOMAIN,
                                               seed=seeds[4])
        chi2_thetas = uniform_random_space(n,
                                           *SPIN_THETA_DOMAIN,
                                            seed=seeds[5])
        chi2_phis = uniform_random_space(n,
                                         *SPIN_PHI_DOMAIN,
                                         seed=seeds[6])
        chi2_spherical_polars = (
            np.hstack([chi2_magnitudes, chi2_thetas, chi2_phis])
            .astype(np.float64)
        )
        chi2_space = spherpol_to_cartesian(chi2_spherical_polars)
        
    elif bbh_spin == "AS":
        q_space = uniform_random_space(n, *MASS_RATIO_DOMAIN, seed=seeds[7])
        chi1_space = uniform_random_space(n, *SPIN_MAG_DOMAIN, seed=seeds[8])
        chi2_space = uniform_random_space(n, *SPIN_MAG_DOMAIN, seed=seeds[9])

    else:
        q_space = uniform_random_space(n, *MASS_RATIO_DOMAIN, seed=seeds[10])
        chi1_space = np.zeros_like(q_space).astype(np.float64)
        chi2_space = np.zeros_like(q_space).astype(np.float64)

    # Combine mass ratio and spins into one param space array, with shape
    # (n, 3) if bbh_spin in {"NS", "AS"} or (n, 7) if bbh_spin is "PS".
    param_space = (
        np.hstack([q_space, chi1_space, chi2_space])
        .astype(np.float64)
    )
    return param_space

HELP_BBH_SPIN = 'Spin configuration: "NS" = no-spin, "AS" = aligned-spin, "PS" = precessing-spin.'
HELP_DATASET = 'Dataset label: "train_xl" = xl training data, "train" = training data, "test" = testing data.'
HELP_SAVING = "Whether to save the parameter space."
HELP_OVERWRITING = "Whether to overwrite a pre-existing parameter space file."
HELP_VERBOSE = "Enable verbose output."

app = typer.Typer(help="Reduced Order Modelling of Gravitational Waves CLI")

@app.command()
def main(
    bbh_spin: BBHSpinType = typer.Option(..., help=HELP_BBH_SPIN),
    dataset: DatasetType = typer.Option(..., help=HELP_DATASET),
    saving: bool = typer.Option(False, help=HELP_SAVING),
    overwriting: bool = typer.Option(False, help=HELP_OVERWRITING),
    verbose: bool = typer.Option(False, help=HELP_VERBOSE),
) -> None:
    """
    Examples
    --------
    Generate and save the no-spin parameter space for the training dataset:

        $ python generate_parameter_space.py --bbh-spin NS --dataset train --saving
    """
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)
    dataset = validate_literal(dataset, DatasetType)
    
    n = N[dataset]
    seeds = SEEDS[dataset]

    if verbose:
        typer.echo(f"Generating parameter space: {bbh_spin=}, {dataset=}")

    param_space = generate_parameter_space(bbh_spin, n, seeds)

    if saving:
        save_dir = PROJECT_ROOT / "data" / bbh_spin / dataset
        save_dir.mkdir(parents=True, exist_ok=True)

        param_space_file = save_dir / "parameter_space.npy"
        if param_space_file.exists() and not overwriting:
            raise FileExistsError(f"File already exists: {param_space_file}")
        np.save(param_space_file, param_space)
    
    if verbose:
        typer.echo(f"Parameter space generation complete.")

if __name__ == "__main__":
    app()
