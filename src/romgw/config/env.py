from __future__ import annotations

from pathlib import Path
import numpy as np
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the RedOrModGW project."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Paths ---
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )

    # --- Numerical constants ---
    zero_tol: float = 1e-14
    common_time_start: float = -5000.0
    common_time_stop: float = 250.0
    common_time_step: float = 1.0

    # --- Raw string environment fields ---
    components_raw: str = "amplitude,phase"
    modes_raw: str = "2,2;2,1;3,3;3,2;4,4;4,3"

    # --- Derived properties ---
    @property
    def common_time(self) -> np.ndarray:
        """Common time grid for interpolation."""
        return np.arange(
            start=self.common_time_start,
            stop=self.common_time_stop,
            step=self.common_time_step,
            dtype=np.float64,
        )

    @property
    def components(self) -> list[str]:
        """List of waveform components (split from env)."""
        return [s.strip() for s in self.components_raw.split(",") if s.strip()]

    @property
    def modes(self) -> list[str]:
        """List of waveform modes (split from env)."""
        return [s.strip() for s in self.modes_raw.split(";") if s.strip()]


# Global singleton
settings = Settings()

# Convenient aliases
# PROJECT_ROOT = settings.project_root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[3]
COMMON_TIME = settings.common_time
ZERO_TOL = settings.zero_tol
