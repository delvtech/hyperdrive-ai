import tempfile
from datetime import datetime
from os import makedirs
from os.path import dirname, exists, realpath
from typing import Callable

from ray.tune.logger import Logger, UnifiedLogger


def create_log_dir(dir: str | None = None, prefix: str = "") -> str:
    """Create log dir with timestamp and return path"""
    this_file = realpath(__file__)
    project_path = dirname(dirname(dirname(this_file)))
    parent_dir = f"{project_path}/checkpoints" if dir is None else dir
    if not exists(parent_dir):
        makedirs(parent_dir, exist_ok=True)

    timestamp = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    prefix = f"{prefix}_" if prefix else prefix
    logdir_prefix = f"{prefix}{timestamp}_"
    logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=parent_dir)
    return logdir


def get_logger_creator(logdir: str) -> Callable[[dict], Logger]:
    """Creates a custom logger factory for use with RLlib."""
    return lambda config: UnifiedLogger(config, logdir, loggers=None)
