from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aiod-utils")
except PackageNotFoundError:
    __version__ = "unknown"

from aiod_utils.preprocess import Preprocess, run_preprocess

__all__ = ["Preprocess", "run_preprocess"]
