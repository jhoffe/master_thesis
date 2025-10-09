import warnings

from transformers import logging as transformers_logging


def ignore_warnings():
    """Ignore specific warnings."""
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    transformers_logging.set_verbosity_error()
