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

# Compare values
OmegaConf.register_new_resolver(
    "equals",
    lambda x, y: x == y,
    replace=True,
)
