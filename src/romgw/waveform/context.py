from __future__ import annotations

from dataclasses import dataclass

# from romgw.typing.utils import validate_literal
from romgw.config.validation import (
    validate_literal,
    validate_dependent_literal
)
# from romgw.typing.core import (
#     BBHSpinType,
#     ModeType,
#     ComponentType,
# )
from romgw.config.types import (
    BBHSpinType,
    ModeType,
    ComponentType,
)
from romgw.config.constants import MODE_VALUES


@dataclass(frozen=True, slots=True)
class Context:
    bbh_spin: BBHSpinType
    mode: ModeType | None = None
    component: ComponentType | None = None

    def __post_init__(self):
        bbh_spin_valid = validate_literal(
            value=self.bbh_spin,
            literal_type=BBHSpinType,
        )
        object.__setattr__(self, "bbh_spin", bbh_spin_valid)

        if self.mode is not None:
            mode_valid = validate_dependent_literal(
                value=self.mode,
                literal_type=ModeType,
                parent_value=self.bbh_spin,
                parent_literal_type=BBHSpinType,
                dependency_map=MODE_VALUES,
            )
            object.__setattr__(self, "mode", mode_valid)

        if self.component is not None:
            component_valid = validate_literal(
                value=self.component,
                literal_type=ComponentType,
            )
            object.__setattr__(self, "component", component_valid)
        

@dataclass(frozen=True, slots=True)
class ComponentContext:
    bbh_spin: BBHSpinType
    mode: ModeType
    component: ComponentType

    def __post_init__(self):
        bbh_spin_valid = validate_literal(
            value=self.bbh_spin,
            literal_type=BBHSpinType,
        )
        object.__setattr__(self, "bbh_spin", bbh_spin_valid)

        mode_valid = validate_dependent_literal(
            value=self.mode,
            literal_type=ModeType,
            parent_value=self.bbh_spin,
            parent_literal_type=BBHSpinType,
            dependency_map=MODE_VALUES,
        )
        object.__setattr__(self, "mode", mode_valid)

        component_valid = validate_literal(
            value=self.component,
            literal_type=ComponentType,
        )
        object.__setattr__(self, "component", component_valid)


@dataclass(frozen=True, slots=True)
class ModeContext:
    bbh_spin: BBHSpinType
    mode: ModeType

    def __post_init__(self):
        bbh_spin_valid = validate_literal(
            value=self.bbh_spin,
            literal_type=BBHSpinType,
        )
        object.__setattr__(self, "bbh_spin", bbh_spin_valid)

        mode_valid = validate_dependent_literal(
            value=self.mode,
            literal_type=ModeType,
            parent_value=self.bbh_spin,
            parent_literal_type=BBHSpinType,
            dependency_map=MODE_VALUES,
        )
        object.__setattr__(self, "mode", mode_valid)


@dataclass(frozen=True, slots=True)
class FullContext:
    bbh_spin: BBHSpinType

    def __post_init__(self):
        bbh_spin_valid = validate_literal(
            value=self.bbh_spin,
            literal_type=BBHSpinType,
        )
        object.__setattr__(self, "bbh_spin", bbh_spin_valid)
