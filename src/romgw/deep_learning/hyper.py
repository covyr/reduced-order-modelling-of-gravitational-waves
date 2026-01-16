from __future__ import annotations
from pathlib import Path
from typing import Literal
import json

class HyperParams:
    optimiser: dict[Literal["epsilon", "learning_rate"], float]
    lrop: dict[Literal["factor", "patience", "min_lr"], float]
    early_stopping: dict[Literal["patience"], int]
    fit: dict[Literal["epochs", "batch_size"], int]

    def __init__(
        self,
        optimiser: dict[Literal["epsilon", "learning_rate"], float] | None = None,
        lrop: dict[Literal["factor", "patience", "min_lr"], float] | None = None,
        early_stopping: dict[Literal["patience"], int] | None = None,
        fit: dict[Literal["epochs", "batch_size"], int] | None = None,
    ):
        if any([optimiser is None, lrop is None, early_stopping is None, fit is None]):
            missing = [hp for hp in [optimiser, lrop, early_stopping, fit] if hp is None]
            raise KeyError(f"Need all of: optimiser, lrop, early_stopping, fit."
                           f"Missing: {', '.join([str(hp) for hp in missing])}")
        
        self.optimiser = dict(**optimiser)
        self.lrop = dict(**lrop)
        self.early_stopping = dict(**early_stopping)
        self.fit = dict(**fit)

    @classmethod
    def from_dir(cls, dir: str | Path) -> HyperParams:
        dir = Path(dir)
        if not dir.is_dir():
            raise NotADirectoryError(f"HyperParams directory does not exist: {dir}")
        data = {}
        for hp in ("optimiser", "lrop", "early_stopping", "fit"):
            with open(dir / f"{hp}.json", 'r') as f:
                data[hp] = dict(**json.load(f))
        return cls(**data)
        
    def to_dir(self, dir: str | Path) -> None:
        dir = Path(dir)
        if not dir.is_dir():
            raise NotADirectoryError(f"HyperParams directory does not exist: {dir}")
        for hp, data in self.__dict__().items():
            with open(dir / f"{hp}.json", 'w') as f:
                json.dump(dict(**data), f)
        
    def __dict__(self) -> dict:
        return {
            "optimiser": dict(**self.optimiser),
            "lrop": dict(**self.lrop),
            "early_stopping": dict(**self.early_stopping),
            "fit": dict(**self.fit),
        }
    
    def __repr__(self) -> str:
        return (
            f"HyperParams:\n"
            f"- optimiser: {', '.join([str(k) + '=' + str(v) for k, v in self.optimiser.items()])}\n"
            f"- lrop: {', '.join([str(k) + '=' + str(v) for k, v in self.lrop.items()])}\n"
            f"- early_stopping: {', '.join([str(k) + '=' + str(v) for k, v in self.early_stopping.items()])}\n"
            f"- fit: {', '.join([str(k) + '=' + str(v) for k, v in self.fit.items()])}"
        )
