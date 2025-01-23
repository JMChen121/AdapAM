from .run import run as default_run
from .masker_run import run as masker_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["masker"] = masker_run
