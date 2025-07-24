from datetime import datetime
from typing import Any

from omegaconf import OmegaConf

# Custom resolvers

# Ternary, requires boolean input
OmegaConf.register_new_resolver(
    "ternary",
    lambda cond, true_val, false_val: (
        true_val
        if cond == True
        else false_val if cond == False else Exception(f"invalid condition ${cond}")
    ),
    replace=True,
)


# table lookup
def _table_resolver(key: Any, *args: Any) -> Any:
    """
    Table lookup resolver.
    Usage: ${table:key,k1,v1,k2,v2,...}
    """
    if len(args) % 2 != 0:
        raise ValueError(
            "table resolver requires an even number of key-value arguments"
        )
    table = dict(zip(args[0::2], args[1::2]))
    return table.get(key)


OmegaConf.register_new_resolver(
    "table",
    _table_resolver,
    replace=True,
)


# Compare values
OmegaConf.register_new_resolver(
    "equals",
    lambda x, y: x == y,
    replace=True,
)
OmegaConf.register_new_resolver(
    "greater_than",
    lambda x, y: x > y,
    replace=True,
)

# now timestamp (used in Public Multiflow, needed to load ckpt)
OmegaConf.register_new_resolver(
    "now",
    lambda fmt: datetime.now().strftime(fmt),
    replace=True,
)
