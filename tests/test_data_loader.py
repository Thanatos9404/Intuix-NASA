import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import pandas as pd
from src.data_loader import map_disposition, load_config


def test_map_disposition():
    config = load_config()

    assert map_disposition("CONFIRMED", config) == "confirmed"
    assert map_disposition("CANDIDATE", config) == "candidate"
    assert map_disposition("FALSE POSITIVE", config) == "false_positive"
    assert map_disposition("FP", config) == "false_positive"
    assert map_disposition(None, config) == "unknown"


def test_config_load():
    config = load_config()
    assert "data" in config
    assert "model" in config
    assert "labels" in config
