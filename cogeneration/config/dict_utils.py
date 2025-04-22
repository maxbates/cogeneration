from dataclasses import fields, is_dataclass
from typing import Any, Dict, List, Tuple, Type


def flatten_dict(raw_dict) -> List[Tuple[str, Any]]:
    """Flattens a nested dict, where nested keys are colon-separated"""
    flattened = []
    for k, v in raw_dict.items():
        if isinstance(v, dict):
            flattened.extend([(f"{k}:{i}", j) for i, j in flatten_dict(v)])
        else:
            flattened.append((k, v))
    return flattened


def deep_merge_dicts(a: Dict, b: Dict) -> Dict:
    """Recursively merge two dictionaries, with values from b overriding a. Assumes compatible structure."""
    result = a.copy()
    for k, v in b.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def prune_unknown_dataclass_fields(
    cls: Type[Any], obj: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively remove keys not defined in the dataclass `cls`."""
    pruned: dict = {}
    for f in fields(cls):
        name = f.name
        if name in obj:
            val = obj[name]
            if is_dataclass(f.type) and isinstance(val, dict):
                pruned[name] = prune_unknown_dataclass_fields(cls=f.type, obj=val)
            else:
                pruned[name] = val
    return pruned
