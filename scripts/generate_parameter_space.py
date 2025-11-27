from numpy.typing import NDArray
import numpy as np

from romgw.config.env import PROJECT_ROOT
from romgw.typing.utils import validate_literal
from romgw.typing.core import RealArray, BBHSpinType, DatasetType


# Sizes of training and test datasets.
N_TRAIN = 4096
N_TEST = 256

# Mass ratio domain over which to sample.
MASS_RATIO_DOMAIN = (1, 10)

# Spherical polar spin component domains over which to sample.
SPIN_MAG_DOMAIN = (-1.0, 1.0)
SPIN_THETA_DOMAIN = (-np.pi/2, np.pi/2)
SPIN_PHI_DOMAIN = (0.0, np.pi)

# Reproducibility.
TRAIN_SEEDS = [19, 29, 59, 89, 229, 521, 599, 1129, 1229, 1259, 2591]
TEST_SEEDS = [2, 73, 179, 283, 419, 547, 661, 811, 947, 1087, 1229]


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


def generate_param_space(
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


def main(
    bbh_spin: BBHSpinType,
    dataset: DatasetType,
    n: int | None = None,
    seeds: list[int] | None = None,
    saving: bool = False,
    overwriting: bool = False,
    verbose: bool = False,
) -> None:
    """"""
    # Validate literals. Raises exception if invalid.
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)
    dataset = validate_literal(dataset, DatasetType)
    
    if not n:
        n = N_TRAIN if dataset == "train" else N_TEST
    
    if not seeds:
        seeds = TRAIN_SEEDS if dataset == "train" else TEST_SEEDS

    if verbose:
        print(f"Generating parameter space for {bbh_spin=}, {dataset=}.")

    param_space = generate_param_space(bbh_spin, n, seeds)

    if saving:
        save_dir = PROJECT_ROOT / "data" / bbh_spin / dataset
        save_dir.mkdir(parents=True, exist_ok=True)

        param_space_file = save_dir / "parameter_space.npy"
        if param_space_file.exists() and not overwriting:
            raise FileExistsError(f"File already exists: {param_space_file}")
        np.save(param_space_file, param_space)
    
    if verbose:
        print(f"Parameter space generation complete.")


if __name__ == "__main__":
    main(bbh_spin="NS",
         dataset="train",
         saving=True,
         overwriting=False,
         verbose=True)
