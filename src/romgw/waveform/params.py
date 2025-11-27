from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from romgw.typing.core import RealArray, MassRatio, Spin
from romgw.typing.utils import validate_mass_ratio, validate_spin


# ----------------------------------------------
# PHYSICAL METADATA
# ----------------------------------------------
@dataclass(frozen=True, slots=True)
class PhysicalParams:
    """
    Physical BBH parameters associated with a waveform.
    
    Attributes
    ----------
    q : MassRatio
        Mass ratio (m1/m2) where m1 >= m2 and q >= 1.
    chi1 : Spin
        Dimensionless spin of the primary BH (scalar or 3-vector).
    chi2 : Spin
        Dimensionless spin of the secondary BH (scalar or 3-vector).
    """
    q: MassRatio
    chi1: Spin
    chi2: Spin

    def __post_init__(self):
        # Validate each field.
        q_valid = validate_mass_ratio(self.q)
        chi1_valid = validate_spin(self.chi1)
        chi2_valid = validate_spin(self.chi2)

        # Enforce dimensional consistency between spins.
        dim1 = 0 if np.ndim(chi1_valid) == 0 else np.size(chi1_valid)
        dim2 = 0 if np.ndim(chi2_valid) == 0 else np.size(chi2_valid)

        if dim1 != dim2:
            raise ValueError(
                f"chi1 and chi2 must have the same dimensionality, "
                f"got chi1: {dim1}-dimensional, chi2: {dim2}-dimensional."
            )

        # Write back validated values.
        object.__setattr__(self, "q", q_valid)
        object.__setattr__(self, "chi1", chi1_valid)
        object.__setattr__(self, "chi2", chi2_valid)

    @property
    def array(self) -> RealArray:
        """Return a 1d array containing the parameter values."""
        params_array = np.concatenate([np.atleast_1d(self.q),
                                       np.atleast_1d(self.chi1),
                                       np.atleast_1d(self.chi2)])
        return params_array

    def __str__(self) -> str:
        _str = lambda x: (f"{x:.6f}" if float(x) != 0 else "0.      ")
        
        q_str = _str(self.q)
        if np.ndim(self.chi1) != 0:
            chi1_str = "[ " + " ".join([_str(x) for x in self.chi1]) + " ]"
            chi2_str = "[ " + " ".join([_str(x) for x in self.chi2]) + " ]"
        
        else:
            chi1_str = (
                "[ " + " ".join([_str(x) for x in [0, 0, self.chi1]]) + " ]"
            )
            chi2_str = (
                "[ " + " ".join([_str(x) for x in [0, 0, self.chi2]]) + " ]"
            )

        return (
            f"PhysicalParams"
            f"(q={q_str}, chi1={chi1_str}, chi2={chi2_str})"
        )
