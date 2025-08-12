import logging
import warnings

from pytorch_lightning.utilities.rank_zero import rank_zero_only


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


# set default log level
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s"
)

# quiet 3rd party loggers
for pkg in [
    "prody",
    "numba",
    "pytorch_lightning",
    "matplotlib.font_manager",
    "matplotlib.animation",
]:
    logging.getLogger(pkg).setLevel(logging.WARNING)
    logging.getLogger(pkg).propagate = False


# quiet warnings
warnings.filterwarnings(
    "ignore",
    message="'pin_memory' argument is set as true but not supported on MPS now, then device pinned memory won't be used.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Consider setting `persistent_workers=True` in 'predict_dataloader' to speed up the dataloader worker initialization.",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="The 'predict_dataloader' does not have many workers which may be a bottleneck.*?",
    category=UserWarning,
)
