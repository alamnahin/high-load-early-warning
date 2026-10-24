"""Runs the output checks of src/04_validate.py - skipped when the data has not been generated."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from config import DATASETS, MODEL_DIR

SRC = Path(__file__).resolve().parents[1] / "src"
have_data = all((MODEL_DIR / f"{ds}_model_ready.parquet").exists() for ds in DATASETS)


def _validate_module():
    spec = importlib.util.spec_from_file_location("validate04", SRC / "04_validate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.skipif(not have_data, reason="run `make all` first to generate data/model_ready")
@pytest.mark.parametrize("ds", list(DATASETS))
def test_generated_tables(ds):
    v = _validate_module()
    mr = pd.read_parquet(MODEL_DIR / f"{ds}_model_ready.parquet")
    v.check_grid(ds)
    v.check_features(ds, mr)
    v.check_targets(ds, mr)
    v.check_split(ds, mr)
    v.check_causality(ds)
