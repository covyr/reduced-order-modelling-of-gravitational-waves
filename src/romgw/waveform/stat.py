from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import json
import numpy as np

from romgw.waveform.params import PhysicalParams
from romgw.typing.core import RealArray, ModeType


# --------------------------------------------------------------------
# ModelStat: timing and parameter info for waveform generation
# --------------------------------------------------------------------
@dataclass
class ModelStat:
    """Timing and parameter statistics for fiducial waveform generation."""
    
    approximant: str
    modes: list[ModeType]
    generation_time: float
    params: PhysicalParams

    # ----- IO -----
    def to_dict(self) -> dict:
        """Convert to a JSON-safe dictionary."""
        d = asdict(self)
        # Ensure numpy arrays are serialised.
        d["params"] = {
            k: (v.tolist()) if isinstance(v, np.ndarray) else v
            for k, v in asdict(self.params).items()
        }
        return d
    
    def to_file(self, path: str | Path) -> None:
        """Save to a .json file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_file(cls, path: str | Path) -> ModelStat:
        """Load from a .json file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r") as f:
            data = json.load(f)

        # Rebuild Physical Params (convert chi1, chi2 back to numpy arrays).
        params_dict = data.pop("params")
        for key in ("chi1", "chi2"):
            val = params_dict[key]
            if isinstance(val, list):
                params_dict[key] = np.array(val, dtype=np.float64)

        params = PhysicalParams(**params_dict)
        return cls(params=params, **data)

    # ----- Representation -----
    def __repr__(self) -> str:
        return (
            f"ModelStat("
            f"approximant={self.approximant!r}, "
            f"modes={self.modes}, "
            f"q={self.params.q}, "
            f"chi1={self.params.chi1}, "
            f"chi2={self.params.chi2}, "
            f"time={self.generation_time:.3f}s)"
        )


# --------------------------------------------------------------------
# ModelStatDataset: multiple ModelStat objects
# --------------------------------------------------------------------
@dataclass
class ModelStatDataset:
    """A collection of ModelStat objects."""

    stats: list[ModelStat] = field(default_factory=list)

    # ----- Core -----
    def add_stat(self, stat: ModelStat) -> None:
        self.stats.append(stat)
    
    def params(self, key: str) -> np.ndarray:
        """Extract one parameter (e.g. 'q', 'chi1', ...) across all stats."""
        return np.array([getattr(stat, key) for stat in self.stats])

    def generation_times(self) -> RealArray:
        """Return array of generation times."""
        return (
            np.array([s.generation_time for s in self.stats])
            .astype(np.float64)
        )

    # ----- IO -----
    @classmethod
    def from_directory(
        cls: ModelStatDataset,
        directory: str | Path
    ) -> ModelStatDataset:
        """Load from a directory containing .json files."""
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        json_files = sorted(directory.glob("*.json"))
        if not json_files:
            raise FileNotFoundError(f"No .json files found in {directory}") 

        lsM = len(str((M := len(json_files)) - 1))
        stats = []
        i = 0
        for json_file in json_files:
            i += 1
            print(
                f"Loading ModelStat "
                f"({i:0{lsM}d}/{len(json_files)})", 
                end='\r'
            )
            
            stat = ModelStat.from_file(json_file)
            stats.append(stat)

        print(' ' * 80, end='\r')
        print("ModelStatDataset loaded.")
        return cls(stats)
    
    # ----- Pythonic interface -----
    def __len__(self):
        return len(self.stats)

    def __iter__(self):
        return iter(self.stats)

    def __repr__(self) -> str:
        if not self.stats:
            return "ModelStatDataset(empty)"
        count = len(self.stats)
        return (
            f"ModelStatDataset<ModelStat>({count} stat"
            f"{'s' if count != 1 else ''})"
        )
