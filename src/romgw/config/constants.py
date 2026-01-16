from __future__ import annotations

from pathlib import Path
import numpy as np


# -----------------------------------------------------------------------------
# ENVIRONMENT VARIABLES
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
COMMON_TIME = np.arange(
    start=-5000,
    stop=251,
    step=1,
    dtype=np.int64,
).astype(np.float64)
ZERO_TOL = 1e-14


# -----------------------------------------------------------------------------
# RUNTIME VALUE SETS
# -----------------------------------------------------------------------------
DATASET_VALUES: set[str] = {"test", "train", "train_xl"}
MODELNAME_VALUES: set[str] = {"NonLinRegV1"}
BBHSPIN_VALUES: set[str] = {"NS", "AS", "PS"}
MODE_VALUES: dict[str, set[str]] = {
    "NS": {"2,2", "2,1", "3,3", "3,2", "4,4", "4,3"},
    "AS": {"2,2", "2,1", "3,3", "3,2", "4,4", "4,3"},
    "PS": {
        "2,2", "2,1", "2,0", "2,-1", "2,-2",
        "3,3", "3,2", "3,1", "3,0", "3,-1", "3,-2", "3,-3",
        "4,4", "4,3", "4,2", "4,1", "4,0", "4,-1", "4,-2", "4,-3", "4,-4"
    }
}
COMPONENT_VALUES: set[str] = {"amplitude", "phase"}
