from __future__ import annotations

from dataclasses import dataclass

from romgw.typing.utils import validate_literal
from romgw.typing.core import (
    BBHSpinType,
    ModeType,
    ComponentType,
)

@dataclass(frozen=True, slots=True)
class ComponentContext:
    bbh_spin: BBHSpinType
    mode: ModeType
    component: ComponentType

    def __post_init__(self):
        bbh_spin_valid = validate_literal(self.bbh_spin, BBHSpinType)
        mode_valid = validate_literal(self.mode, ModeType)
        component_valid = validate_literal(self.component, ComponentType)

        object.__setattr__(self, "bbh_spin", bbh_spin_valid)
        object.__setattr__(self, "mode", mode_valid)
        object.__setattr__(self, "component", component_valid)
        

@dataclass(frozen=True, slots=True)
class ModeContext:
    bbh_spin: BBHSpinType
    mode: ModeType

    def __post_init__(self):
        bbh_spin_valid = validate_literal(self.bbh_spin, BBHSpinType)
        mode_valid = validate_literal(self.mode, ModeType)

        object.__setattr__(self, "bbh_spin", bbh_spin_valid)
        object.__setattr__(self, "mode", mode_valid)


@dataclass(frozen=True, slots=True)
class FullContext:
    bbh_spin: BBHSpinType

    def __post_init__(self):
        bbh_spin_valid = validate_literal(self.bbh_spin, BBHSpinType)
        object.__setattr__(self, "bbh_spin", bbh_spin_valid)
