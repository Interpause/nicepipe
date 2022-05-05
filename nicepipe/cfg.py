from __future__ import annotations
from dataclasses import dataclass, field
import logging
import os

from omegaconf import OmegaConf

from .worker import workerCfg

log = logging.getLogger(__name__)


@dataclass
class miscCfg:
    skip_tests: bool = False
    log_level: int = logging.INFO


@dataclass
class nicepipeCfg:
    worker: workerCfg = field(default_factory=workerCfg)
    misc: miscCfg = field(default_factory=miscCfg)


# TODO: actually merge defaults and accept CLI overrides
def get_config(path="config.yml"):
    """
    Get config file. If it doesn't exist, it will create the default and throw
    KeyboardInterrupt. If invalid, it will throw ValidationError.
    """
    schema = OmegaConf.structured(nicepipeCfg)
    if not os.path.exists(path):
        log.warning(f"{path} not found! creating...")
        OmegaConf.save(schema, path)
        log.warning(f"Program will now exit to allow you to edit {path}.")
        raise KeyboardInterrupt

    cfg = OmegaConf.unsafe_merge(schema, OmegaConf.load(path))
    log.debug(
        f"Config:\n{cfg}",
    )
    return cfg
