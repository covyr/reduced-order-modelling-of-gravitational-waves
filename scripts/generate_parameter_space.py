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

    # Handle NoneType seeds.
    if not seeds:
        seeds = [None] * 11
    
    # Generate param space for "PS" BBHSpinType.
    if bbh_spin == "PS":
        # Mass ratios.
        q_space = uniform_random_space(n,
                                       seed=seeds[0],
                                       *MASS_RATIO_DOMAIN)
        
        # Spherial polar components of chi1.
        chi1_mag_space = uniform_random_space(n,
                                              seed=seeds[1],
                                              *SPIN_MAG_DOMAIN)
        chi1_theta_space = uniform_random_space(n,
                                                seed=seeds[2],
                                                *SPIN_THETA_DOMAIN)
        chi1_phi_space = uniform_random_space(n,
                                              seed=seeds[3],
                                              *SPIN_PHI_DOMAIN)
        # Combine spherical polar components of chi1 into one array.
        chi1_spherical_polars = (
            np.hstack([chi1_mag_space, chi1_theta_space, chi1_phi_space])
            .astype(np.float64)
        )
        # Convert chi1 spherical polar components to cartesian components.
        chi1_space = spherpol_to_cartesian(chi1_spherical_polars)

        # Spherial polar components of chi2.
        chi2_mag_space = uniform_random_space(n,
                                              seed=seeds[4],
                                              *SPIN_MAG_DOMAIN)
        chi2_theta_space = uniform_random_space(n,
                                                seed=seeds[5],
                                                *SPIN_THETA_DOMAIN)
        chi2_phi_space = uniform_random_space(n,
                                              seed=seeds[6],
                                              *SPIN_PHI_DOMAIN)
        # Combine spherical polar components of chi1 into one array.
        chi2_spherical_polars = (
            np.hstack([chi2_mag_space, chi2_theta_space, chi2_phi_space])
            .astype(np.float64)
        )
        # Convert chi1 spherical polar components to cartesian components.
        chi2_space = spherpol_to_cartesian(chi2_spherical_polars)
        
    # Generate param space for "AS" BBHSpinType.
    elif bbh_spin == "AS":
        # Mass ratios.
        q_space = uniform_random_space(n,
                                       seed=seeds[7],
                                       *MASS_RATIO_DOMAIN)
        # Scalar chi1.
        chi1_space = uniform_random_space(n,
                                          seed=seeds[8],
                                          *SPIN_MAG_DOMAIN)
        # Scalar chi2.
        chi2_space = uniform_random_space(n,
                                          seed=seeds[9],
                                          *SPIN_MAG_DOMAIN)

    # Generate param space for "NS" BBHSpinType.
    else:
        # Mass ratios.
        q_space = uniform_random_space(n,
                                       seed=seeds[10],
                                       *MASS_RATIO_DOMAIN)
        # Zero chi1.
        chi1_space = np.zeros_like(q_space).astype(np.float64)
        # Zero chi2.
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
    overwrite: bool = False,
) -> None:
    """"""
    # Validate literals. Raises exception if invalid.
    bbh_spin = validate_literal(bbh_spin, BBHSpinType)
    dataset = validate_literal(dataset, DatasetType)
    
    # Auto-assign n to the constant corresponding to the dataset.
    if not n:
        n = N_TRAIN if dataset == "train" else N_TEST
    
    # Auto-assign seeds to the constants corresponding to the dataset.
    if not seeds:
        seeds = TRAIN_SEEDS if dataset == "train" else TEST_SEEDS

    # Generate param space.
    print(f"Generating parameter space for {bbh_spin=}, {dataset=}.")

    param_space = generate_param_space(bbh_spin, n, seeds)

    if saving:
        # Directory to save the param space into.
        data_dir = PROJECT_ROOT / "data" / bbh_spin / dataset
        # Ensure directory exists.
        data_dir.mkdir(parents=True, exist_ok=True)

        param_space_file = data_dir / "parameter_space.npy"
        if param_space_file.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {param_space_file}")
        # Save param space.
        np.save(param_space_file, param_space)
            
    print(f"Parameter space generation complete.")


if __name__ == "__main__":
    main(bbh_spin="NS",
         dataset="test",
         saving=False,
         overwrite=False)
