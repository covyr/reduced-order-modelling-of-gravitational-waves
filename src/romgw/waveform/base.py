from __future__ import annotations

from numpy.typing import NDArray
from pathlib import Path
import numpy as np

from romgw.waveform.params import PhysicalParams
from romgw.typing.core import (
    RealArray,
    ComplexArray,
    ComponentType,
    Spin,
)


# ----------------------------------------------
# BASE WAVEFORM CLASS
# ----------------------------------------------
class BaseWaveform(np.ndarray):
    """Base waveform class with physical params."""

    params: PhysicalParams

    # ----- Constructors -----
    def __new__(
        cls,
        waveform_arr: NDArray,
        params: PhysicalParams,
    ) -> "BaseWaveform":
        obj = np.asarray(waveform_arr).view(cls)
        obj.params = params
        return obj
    
    def __array_finalize__(self, obj: NDArray | None = None) -> None:
        if obj is None:
            return
        self.params = getattr(obj, "params", getattr(self, "params", None))

    # ----- IO -----
    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "BaseWaveform":
        """Load waveform and params."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with np.load(path, allow_pickle=False) as f:
            data = f["data"]
            q = float(f["q"])

            def _restore_spin(arr: np.ndarray) -> Spin:
                if np.ndim(arr) == 0:  # scalar spin
                    return float(arr)
                return arr.astype(np.float64)

            chi1 = _restore_spin(f["chi1"])
            chi2 = _restore_spin(f["chi2"])

        params = PhysicalParams(q=q, chi1=chi1, chi2=chi2)
        return cls(data, params, **kwargs)
    
    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        seed: int | None = None,
        **kwargs
    ) -> "BaseWaveform":
        """Load a random waveform from a directory."""
        seed = seed if seed else None
        
        path = Path(path)
        if not path.exists():
            raise NotADirectoryError(f"Directory not found: {path}")
        
        files = sorted(path.glob("*.npz"))
        if not files:
            raise ValueError(f"No waveform files found in directory: {path}")
        
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(files))[0]
        return cls._load_waveform(files[idx], **kwargs)
    
    def to_file(self, path: str | Path, overwrite: bool = False) -> None:
        """Save waveform and params."""
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"File already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez(
            path,
            data=np.asarray(self, dtype=self.dtype),
            q=np.float64(self.params.q),
            chi1=np.asarray(self.params.chi1, dtype=np.float64),
            chi2=np.asarray(self.params.chi2, dtype=np.float64),
        )

    # ----- Internal helpers -----
    @staticmethod
    def _load_waveform(path: str | Path, **kwargs) -> BaseWaveform:
        """Override in subclass for type-specific loading."""
        raise NotImplementedError("Must be implemented in subclass.")

    # ----- Representation -----
    def __repr__(self) -> str:
        p = self.params
        return (
            f"<{self.__class__.__name__}(q={p.q:.2f}, "
            f"chi1={p.chi1}, chi2={p.chi2}), "
            f"shape={self.shape}, dtype={self.dtype}>"
        )
    

# ----------------------------------------------
# COMPONENT WAVEFORM (Amplitude / Phase)
# ----------------------------------------------
class ComponentWaveform(BaseWaveform):
    """Real-valued waveform component (amplitude or phase)."""

    component: ComponentType

    # ----- Constructors -----
    def __new__(
        cls,
        waveform_arr: RealArray,
        params: PhysicalParams,
        component: ComponentType,
    ) -> "ComponentWaveform":
        obj = super().__new__(cls, waveform_arr, params)
        obj.component = component
        if not np.issubdtype(obj.dtype, np.floating):
            raise TypeError(f"{component} must be real-valued.")
        return obj

    def __array_finalize__(self, obj: RealArray | None = None) -> None:
        super().__array_finalize__(obj)
        if obj is None:
            return
        self.component = getattr(obj,
                                 "component",
                                 getattr(self, "component", None))

    @staticmethod
    def _load_waveform(
        path: str | Path,
        **kwargs,
    ) -> ComponentWaveform:
        """"""
        path = Path(path)
        return ComponentWaveform.from_file(path, **kwargs)

    # ----- Representation -----
    def __repr__(self) -> str:
        p = self.params
        return (
            f"<ComponentWaveform[{self.component}]("
            f"q={p.q:.2f}, chi1={p.chi1}, chi2={p.chi2}), "
            f"shape={self.shape}>"
        )


# ----------------------------------------------
# FULL COMPLEX WAVEFORM
# ----------------------------------------------
class FullWaveform(BaseWaveform):
    """Complex-valued full waveform."""

    # ----- Constructors -----
    def __new__(
        cls,
        waveform_arr: ComplexArray,
        params: PhysicalParams,
    ) -> "FullWaveform":
        obj = super().__new__(cls, waveform_arr, params)
        if not np.issubdtype(obj.dtype, np.complexfloating):
            raise TypeError("FullWaveform must be complex-valued.")
        return obj
    
    # ----- Alternative constructors -----
    @classmethod
    def from_components(
        cls,
        amplitude: ComponentWaveform,
        phase: ComponentWaveform,
    ) -> "FullWaveform":
        """Recombine amplitude and phase into a full complex waveform."""
        
        if (
            not isinstance(amplitude, ComponentWaveform) 
            or not isinstance(phase, ComponentWaveform)
        ):
            raise TypeError("Inputs must be ComponentWaveform instances.")

    
        if amplitude.component != "amplitude" or phase.component != "phase":
            raise ValueError("Must provide amplitude and phase "
                             "components respectively.")
        # Check shape consistency. 
        if amplitude.shape != phase.shape:
            raise ValueError("Amplitude and phase must have the same shape.")

        # Check parameter consistency.
        if amplitude.params != phase.params:
            raise ValueError(
                f"Amplitude and phase parameter mismatch:\n"
                f"{amplitude.params=}\n"
                f"{phase.params=}\n"
                f"Amplitude and phase must be components of the same waveform."
            )

        waveform_arr: ComplexArray = (
            np.asarray(amplitude) * np.asarray(np.exp(-1j * phase))
        )
        params = amplitude.params  # identical params
        return cls(waveform_arr, params)
    
    # ----- Properties -----
    @property
    def amplitude(self) -> ComponentWaveform:
        wf_arr = np.abs(np.asarray(self)).astype(np.float64)
        return ComponentWaveform(wf_arr,
                                 params=self.params,
                                 component="amplitude")

    @property
    def phase(self) -> ComponentWaveform:
        wf_arr = -np.unwrap(np.angle(np.asarray(self))).astype(np.float64)
        return ComponentWaveform(wf_arr,
                                 params=self.params,
                                 component="phase")

    @staticmethod
    def _load_waveform(path: str | Path) -> FullWaveform:
        path = Path(path)
        return FullWaveform.from_file(path)
