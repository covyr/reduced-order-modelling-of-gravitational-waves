from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator
import numpy as np

from romgw.waveform.params import PhysicalParams
# from romgw.typing.core import ComponentType, RealArray
from romgw.config.types import ComponentType, RealArray
from romgw.waveform.base import (
    BaseWaveform,
    ComponentWaveform,
    FullWaveform,
)


# ------------------------------------------------------
# BASE WAVEFORM DATASET
# ------------------------------------------------------
class BaseWaveformDataset:
    """
    Base class for waveform datasets (component or full).
    Provides storage, iteration, filtering, and disk IO utilities.
    """

    def __init__(self, waveforms: Iterable[BaseWaveform]):
        # waveforms = list(waveforms)
        # if not waveforms:
        #     raise ValueError("Dataset cannot be empty.")
        # self._waveforms: list[BaseWaveform] = waveforms
        
        waveforms = list(waveforms) or []
        self._waveforms = waveforms

    def add_waveform(self, wf: BaseWaveform) -> None:
        self._waveforms.append(wf)

    def remove_waveform(self, idx: int) -> None:
        del self._waveforms[idx]

    def pop(self, idx: int | None = None) -> BaseWaveform:
        return self._waveforms.pop(idx)

    # ----- Basic accessors -----
    def __len__(self) -> int:
        return len(self._waveforms)

    def __getitem__(self, idx: int) -> BaseWaveform:
        return self._waveforms[idx]

    def __iter__(self) -> Iterator[BaseWaveform]:
        return iter(self._waveforms)

    # ----- Properties -----
    @property
    def shape(self) -> tuple[int]:
        """Shape of the dataset."""
        return (len(self._waveforms), len(self._waveforms[0]))

    @property
    def params_list(self) -> list[PhysicalParams]:
        """List of PhysicalParams across all waveforms."""
        return [wf.params for wf in self._waveforms]
    
    @property
    def array(self) -> RealArray:
        """"""
        arr = np.vstack([np.asarray(wf) for wf in self._waveforms])
        return arr.astype(np.float64)

    @property
    def params_array(self) -> RealArray:
        """Array of PhysicalParam values across all waveforms."""
        params_list = []
        for wf in self._waveforms:
            params = wf.params
            wf_params = np.concatenate([np.atleast_1d(params.q),
                                        np.atleast_1d(params.chi1),
                                        np.atleast_1d(params.chi2)])
            params_list.append(wf_params)
        
        return np.vstack(params_list).astype(np.float64)

    # ----- IO -----
    def to_directory(self, path: str | Path, overwrite: bool = False) -> None:
        """
        Save all waveforms in the dataset to a directory.
        Each waveform is stored as a separate `.npz` file.
        """
        path = Path(path)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Directory already exists: {path}")
        path.mkdir(parents=True, exist_ok=True)

        lsM = len(str(len(self._waveforms) - 1))
        for i, wf in enumerate(self._waveforms):
            fname = path / f"waveform_{i:0{lsM}d}.npz"
            wf.to_file(fname)

    @classmethod
    def from_directory(
        cls,
        path: str | Path,
        n: int | None = None,
        seed: int | None = None,
        **kwargs
    ) -> BaseWaveformDataset:
        """
        Load all waveforms in a directory into a dataset.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        files = sorted(path.glob("*.npz"))
        if not files:
            raise ValueError(f"No waveform files found in directory: {path}")

        lsM = len(str((M := len(files)) - 1))
        waveforms = []

        loading_n_random = False if not n else True

        if not loading_n_random:
            for i, f in enumerate(files):
                print(f"Loading waveform {i+1:0{lsM}d}/{M}", end='\r')
                waveforms.append(cls._load_waveform(f, **kwargs))
    
        else:  # if loading_n_random
            rng = np.random.default_rng(seed)
            idxs = rng.permutation(M)[:n]
            for i, idx in enumerate(idxs):
                print(f"Loading waveform {i+1:0{lsM}d}/{n}", end='\r')
                waveforms.append(cls._load_waveform(files[idx], **kwargs))

        print(' ' * 80, end='\r')
        print("Waveforms loaded.")
        return cls(waveforms, **kwargs)

    # ----- Internal helpers -----
    @staticmethod
    def _load_waveform(path: str | Path, **kwargs) -> BaseWaveform:
        """Override in subclass for type-specific loading."""
        raise NotImplementedError("Must be implemented in subclass.")


# ------------------------------------------------------
# COMPONENT WAVEFORM DATASET
# ------------------------------------------------------
class ComponentWaveformDataset(BaseWaveformDataset):
    """Dataset of amplitude or phase component waveforms."""

    component: ComponentType

    def __init__(
        self,
        # waveforms: Iterable[ComponentWaveform],
        waveforms: Iterable[ComponentWaveform] | None = None,
        component: ComponentType = "n/a",
    ):
        """"""
        waveforms = list(waveforms) or []
        self.component = component

        super().__init__(waveforms)

    @staticmethod
    def _load_waveform(
        path: str | Path,
        **kwargs,
    ) -> ComponentWaveform:
        """"""
        path = Path(path)
        return ComponentWaveform.from_file(path, **kwargs)

    def __repr__(self) -> str:
        return (
            f"<ComponentWaveformDataset[{self.component}]("
            f"n={self.shape[0]}, L={self.shape[1]})>"
        )


# ------------------------------------------------------
# FULL WAVEFORM DATASET
# ------------------------------------------------------
class FullWaveformDataset(BaseWaveformDataset):
    """Dataset of full complex-valued waveforms."""

    def __init__(self, waveforms: Iterable[FullWaveform]):
        waveforms = list(waveforms)
        if not all(isinstance(wf, FullWaveform) for wf in waveforms):
            raise TypeError("All waveforms must be FullWaveform instances.")
        super().__init__(waveforms)

    @staticmethod
    def _load_waveform(path: str | Path) -> FullWaveform:
        path = Path(path)
        return FullWaveform.from_file(path)

    def __repr__(self) -> str:
        return f"<FullWaveformDataset(n={self.shape[0]}, L={self.shape[1]})>"
