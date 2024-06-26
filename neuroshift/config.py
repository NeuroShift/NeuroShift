"""This module contains the Configurations of the NeuroShift Dashboard."""

from typing import List, Any

import toml
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_conf(path: str) -> None:
    """
    Load the configuration from the path to a given toml file

    Args:
        path (str): the path to the toml configuration file
    """
    try:
        conf = toml.load(path)
    except Exception:  # noqa: every exception could be thrown here
        return

    if "neuroshift" not in conf:
        return

    conf = conf["neuroshift"]

    def set_attr(name: str, value: Any) -> None:
        name = name.upper()
        if name in globals():
            globals()[name] = value

    for k, v in conf.items():
        match k:
            case "paths":
                for name, val in v.items():
                    set_attr(f"{name}_path", val)
            case "settings":
                for name, val in v.items():
                    set_attr(f"{name}_settings", val)
            case "filetypes":
                for name, val in v.items():
                    set_attr(f"allowed_{name}_filetypes", val)
            case _:
                set_attr(k, v)

    global config_file  # noqa
    config_file = path


config_file: str | None = "neuroconf.toml"
MAX_RETRIES: int = 3
BATCH_SIZE: int = 32
WORKERS: int = 3
ANALYTICS_PATH: str = "data/analytics/"
DATASET_PATH: str = "data/datasets/"
DATASET_SETTINGS: str = "datasets.json"
MODEL_PATH: str = "data/models/"
MODEL_SETTINGS: str = "models.json"
ALLOWED_UPLOAD_FILETYPES: List[str] = ["ONNX", "ZIP", "JSON"]
ALLOWED_MODEL_FILETYPES: List[str] = [".onnx"]
ALLOWED_IMAGE_FILETYPES: List[str] = [".jpg", ".jpeg", ".png"]
ANALYTIC_NAMES: List[str] = ["key"]
MAX_WIDTH: int = 500

if config_file is not None:
    load_conf(config_file)
