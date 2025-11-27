from typing import get_args, Any

def literal_values(literal_type: Any) -> set[str]:
    """Return all allowed values of a Literal type."""
    try:
        return set(get_args(literal_type))
    except Exception:
        return []
