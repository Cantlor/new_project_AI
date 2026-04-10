# Logging setup.
# Thin wrapper over the standard library so all callers use a consistent interface.
# (REPO_CONVENTIONS.md §16)

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module or component name.

    Callers should pass a dotted name that reflects module / stage context,
    e.g. "ai_fields.module_prep_data.01_check_inputs".
    (REPO_CONVENTIONS.md §16)
    """
    return logging.getLogger(name)
