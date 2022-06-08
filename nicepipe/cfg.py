from __future__ import annotations
from dataclasses import dataclass, field
import logging
from pathlib import Path

from omegaconf import OmegaConf, SCMode

from .output import outputCfg
from .input import cv2CapCfg
from .analyze import analysisCfg

log = logging.getLogger(__name__)


@dataclass
class miscCfg:
    skip_tests: bool = False
    log_level: int = logging.INFO
    # TODO: disabled until we can get the real CWD in child processes on windows
    save_logs: bool = False
    console_live_display: bool = True
    headless_mode: bool = False


@dataclass
class nicepipeCfg:
    analyze: analysisCfg = field(default_factory=analysisCfg)
    input: cv2CapCfg = field(default_factory=cv2CapCfg)
    output: outputCfg = field(default_factory=outputCfg)
    misc: miscCfg = field(default_factory=miscCfg)


# TODO: Configuration System
# copy template config folder to parent directory
# add template config folder to search path
# compose app config using both builtin and external config groups

# TODO: actually merge defaults and accept CLI overrides
def get_config(path="config.yml"):
    """
    Get config file. If it doesn't exist, it will create the default and throw
    KeyboardInterrupt. If invalid, it will throw ValidationError.
    """
    schema = OmegaConf.create(nicepipeCfg)
    # https://github.com/omry/omegaconf/issues/910
    schema = OmegaConf.to_container(schema, structured_config_mode=SCMode.DICT)
    # OmegaConf.set_struct(schema, False)
    if not Path(path).is_file():
        print(f"{path} not found! creating...")
        OmegaConf.save(schema, path)
        print(
            f"{path} contains all config options and their default values. "
            "At runtime, it is merged with the default config, meaning that "
            f"parts of {path} can be deleted if so desired. For example, a "
            f"blank {path} will result in the default config being used. "
            f"Delete {path} to regenerate the full config file again. "
        )
        print(
            "The config system uses OmegaConf. See "
            "https://omegaconf.readthedocs.io/en/2.1_branch/index.html "
            "for special features available."
        )
        print(f"Program will now exit to allow you to edit {path}.")
        raise KeyboardInterrupt

    cfg = OmegaConf.unsafe_merge(schema, OmegaConf.load(path))
    return cfg
