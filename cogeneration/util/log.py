import logging

from pytorch_lightning.utilities.rank_zero import rank_zero_only

# quiet 3rd party loggers
for pkg in ["prody", "numba", "pytorch_lightning"]:
    logging.getLogger(pkg).setLevel(logging.WARNING)
    logging.getLogger(pkg).propagate = False


def rank_zero_logger(name=__name__) -> logging.Logger:
    """Initializes multi-thread-friendly python logger."""

    logger = logging.getLogger(name)

    # wrap each level with rank_zero_only
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    for level in logging_levels:
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger
