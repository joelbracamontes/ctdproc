"""Tests for processing.validate_features and the two enum-style ValueErrors
in pipeline.apply_bin (unknown bin_type / type_profile).

validate_features exists to catch a misspelled `features:` step name before
any processing starts - without it, an unmatched name just silently skips
that step in every dispatch chain (main.py, cellTM_finder.py,
slopecorrection_finder.py), with no error and nothing in the output to say
a step was ever skipped.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ctdproc.processing import validate_features
from ctdproc.pipeline import apply_bin

VALID_MAIN_STEPS = [
    "align", "low_filter", "wild_edit", "celltm", "slope_corr", "loop_edit", "bin",
]


def test_validate_features_accepts_all_known_steps():
    features = [{"name": name} for name in VALID_MAIN_STEPS]
    validate_features(features, VALID_MAIN_STEPS)  # should not raise


def test_validate_features_accepts_empty_or_missing():
    validate_features([], VALID_MAIN_STEPS)
    validate_features(None, VALID_MAIN_STEPS)


def test_validate_features_rejects_misspelled_step():
    features = [{"name": "align"}, {"name": "wild_eddit"}]
    with pytest.raises(ValueError, match="wild_eddit"):
        validate_features(features, VALID_MAIN_STEPS)


def test_validate_features_lists_all_valid_options_with_meaning():
    with pytest.raises(ValueError) as excinfo:
        validate_features([{"name": "bogus"}], VALID_MAIN_STEPS)
    message = str(excinfo.value)
    for name in VALID_MAIN_STEPS:
        assert name in message


def test_validate_features_reports_every_bad_name_at_once():
    features = [{"name": "align"}, {"name": "bogus_one"}, {"name": "bogus_two"}]
    with pytest.raises(ValueError) as excinfo:
        validate_features(features, VALID_MAIN_STEPS)
    message = str(excinfo.value)
    assert "bogus_one" in message
    assert "bogus_two" in message


def test_validate_features_rejects_subset_valid_for_finder():
    # cellTM_finder only dispatches align/low_filter/wild_edit - celltm itself
    # is a real main.py step name, but not valid *inside* that finder's own
    # features: loop (it's driven by celltm_settings instead).
    with pytest.raises(ValueError, match="celltm"):
        validate_features([{"name": "celltm"}], ["align", "low_filter", "wild_edit"])


def test_apply_bin_unknown_bin_type_lists_valid_options():
    import pandas as pd
    data = pd.DataFrame({"pressure": [1.0, 2.0], "elapsed_time": [0.0, 1.0]})
    step = {"bin_type": ["bogus"], "bin_size": [1.0], "type_profile": ["BOTH"]}
    with pytest.raises(ValueError) as excinfo:
        apply_bin(data, step, bad_flag_value=-9.99e-29, base_name="test")
    message = str(excinfo.value)
    assert "elapsed_time" in message and "scans" in message and "pressure" in message


def test_apply_bin_unknown_type_profile_lists_valid_options():
    import pandas as pd
    data = pd.DataFrame({"pressure": [1.0, 2.0], "elapsed_time": [0.0, 1.0]})
    step = {"bin_type": ["pressure"], "bin_size": [1.0], "type_profile": ["SIDEWAYS"]}
    with pytest.raises(ValueError) as excinfo:
        apply_bin(data, step, bad_flag_value=-9.99e-29, base_name="test")
    message = str(excinfo.value)
    assert "BOTH" in message and "UP" in message and "DOWN" in message
