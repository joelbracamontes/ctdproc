"""Pipeline regression tests against committed reference fixtures.

Per-stage tests: load the reference output from the previous stage, apply
the next stage, compare to the reference CSV for that stage.

Integration test: run the full CTD_process on the fixture cast and compare
the written .cnv files to the reference .cnv files.

Fixtures live in tests/fixtures/reference/.  To regenerate them after an
intentional change, run:

    python tests/generate_fixtures.py

then re-commit the updated reference files.
"""

import sys
import warnings
from io import StringIO
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).parent.parent
FIXTURES = Path(__file__).parent / "fixtures"
RAW_DIR = FIXTURES / "raw"
REF_DIR = FIXTURES / "reference"

sys.path.insert(0, str(ROOT / "src"))

from ctdproc.load import convert_data, load_raw_data
from ctdproc.pipeline import (
    apply_align,
    apply_bin,
    apply_celltm,
    apply_loop_edit,
    apply_low_filter,
    apply_slope_correction,
    apply_wild_edit,
)
from ctdproc.processing import (
    alp_tau,
    alp_tau_fast,
    bottle_avg,
    crosshigh,
    filt_interp,
    find_opt_alp_tat,
    find_opt_alp_tat_fast,
    slope_for_correction,
)
from ctdproc.utilities import bl_reader

CAST = "MIXSED2_000"
SAMPLE_DT = 1 / 24
BAD_FLAG = np.float64(-9.99e-29)

with open(FIXTURES / "config.yaml") as _f:
    FEATURES = yaml.safe_load(_f)["features"]

STEPS = {s["name"]: s for s in FEATURES}


def load_ref(name: str) -> pd.DataFrame:
    return pd.read_csv(REF_DIR / f"{name}.csv.gz")


def assert_df_close(actual: pd.DataFrame, ref: pd.DataFrame, stage: str) -> None:
    assert list(actual.columns) == list(ref.columns), (
        f"{stage}: column mismatch — got {list(actual.columns)}, expected {list(ref.columns)}"
    )
    assert len(actual) == len(ref), (
        f"{stage}: row count mismatch — got {len(actual)}, expected {len(ref)}"
    )
    for col in ref.columns:
        a = actual[col].to_numpy(dtype=float)
        r = ref[col].to_numpy(dtype=float)
        np.testing.assert_array_equal(
            np.isnan(a),
            np.isnan(r),
            err_msg=f"{stage} column '{col}': NaN positions differ",
        )
        mask = ~np.isnan(r)
        np.testing.assert_allclose(
            a[mask],
            r[mask],
            rtol=1e-5,
            atol=1e-9,
            err_msg=f"{stage} column '{col}' differs from reference",
        )


# ---------------------------------------------------------------------------
# Module-level fixtures (shared across all tests)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stage_convert():
    raw = load_raw_data(RAW_DIR / f"{CAST}.hex")
    return convert_data(raw, RAW_DIR / f"{CAST}.xmlcon", SAMPLE_DT)


@pytest.fixture(scope="module")
def stage_align(stage_convert):
    data = stage_convert.copy()
    data, _ = apply_align(data, STEPS["align"], SAMPLE_DT, BAD_FLAG)
    return data


@pytest.fixture(scope="module")
def stage_wild_edit(stage_align):
    data = stage_align.copy()
    data, _ = apply_wild_edit(data, STEPS["wild_edit"], BAD_FLAG)
    return data


@pytest.fixture(scope="module")
def stage_low_filter(stage_wild_edit):
    data = stage_wild_edit.copy()
    data, _ = apply_low_filter(data, STEPS["low_filter"], SAMPLE_DT)
    return data


@pytest.fixture(scope="module")
def stage_celltm(stage_low_filter):
    data = stage_low_filter.copy()
    data, _ = apply_celltm(data, STEPS["celltm"], SAMPLE_DT, BAD_FLAG)
    return data


@pytest.fixture(scope="module")
def stage_loop_edit(stage_celltm):
    data = stage_celltm.copy()
    data, _ = apply_loop_edit(data, STEPS["loop_edit"], SAMPLE_DT, BAD_FLAG)
    return data


# ---------------------------------------------------------------------------
# Per-stage regression tests
# ---------------------------------------------------------------------------


def test_convert(stage_convert):
    assert_df_close(stage_convert, load_ref("stage_00_convert"), "convert_data")


def test_align(stage_align):
    assert_df_close(stage_align, load_ref("stage_01_align"), "align")


def test_wild_edit(stage_wild_edit):
    assert_df_close(stage_wild_edit, load_ref("stage_02_wild_edit"), "wild_edit")


def test_low_filter(stage_low_filter):
    assert_df_close(stage_low_filter, load_ref("stage_03_low_filter"), "low_filter")


def test_celltm(stage_celltm):
    assert_df_close(stage_celltm, load_ref("stage_04_celltm"), "celltm")


def test_loop_edit(stage_loop_edit):
    assert_df_close(stage_loop_edit, load_ref("stage_05_loop_edit"), "loop_edit")


def test_bin_scans(stage_loop_edit):
    data = stage_loop_edit.copy()
    bin_results, _, names = apply_bin(data, STEPS["bin"], BAD_FLAG, CAST)
    label = Path(names[0]).stem
    assert_df_close(bin_results[0], load_ref(f"stage_06_bin_{label}"), f"bin/{label}")


def test_bin_pressure(stage_loop_edit):
    data = stage_loop_edit.copy()
    bin_results, _, names = apply_bin(data, STEPS["bin"], BAD_FLAG, CAST)
    label = Path(names[1]).stem
    assert_df_close(bin_results[1], load_ref(f"stage_06_bin_{label}"), f"bin/{label}")


# ---------------------------------------------------------------------------
# Integration test — full CTD_process → compare .cnv files
# ---------------------------------------------------------------------------


def _load_cnv_data(cnv_path: Path) -> pd.DataFrame:
    """Parse a .cnv file written by main.py - either the SBE-style full
    resolution output ('# name N = LABEL: ...' header, data after '*END*',
    whitespace-separated) or the plain CSV the bin branch still writes
    (comma-separated with its own pandas header row, '*'-prefixed metadata)."""
    lines = cnv_path.read_text(encoding="utf-8").splitlines()

    sbe_columns = []
    data_start = None
    for i, line in enumerate(lines):
        if line.startswith("# name"):
            label = line.split("=", 1)[1].strip().split(":", 1)[0].strip()
            sbe_columns.append(label)
        elif line.startswith("*END*"):
            data_start = i + 1
            break

    if data_start is not None:
        data_text = "\n".join(lines[data_start:])
        return pd.read_csv(StringIO(data_text), sep=r"\s+", header=None, names=sbe_columns)

    data_lines = [l for l in lines if not l.startswith("*")]
    return pd.read_csv(StringIO("\n".join(data_lines)))


def test_full_pipeline_cnv_outputs(tmp_path):
    from ctdproc.main import CTD_process

    cfg_template = (FIXTURES / "config.yaml").read_text()
    cfg_text = cfg_template.replace("__RAW_DIR__", RAW_DIR.resolve().as_posix())
    cfg_text = cfg_text.replace("__OUT_DIR__", tmp_path.resolve().as_posix())

    tmp_cfg = tmp_path / "config.yaml"
    tmp_cfg.write_text(cfg_text)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CTD_process(str(tmp_cfg))

    expected_files = [
        f"{CAST}_24hz.cnv",
        f"{CAST}_24scans.cnv",
        f"down{CAST}_1dbar.cnv",
    ]

    for fname in expected_files:
        out_path = tmp_path / fname
        ref_path = REF_DIR / fname
        assert out_path.exists(), f"pipeline did not produce {fname}"
        assert_df_close(_load_cnv_data(out_path), _load_cnv_data(ref_path), fname)


# ---------------------------------------------------------------------------
# filt_interp
# ---------------------------------------------------------------------------


def test_filt_interp_flag_true_restores_bad_values(stage_convert):
    """With flag=True, injected bad-flag positions must survive filtering."""
    data = stage_convert.copy()
    bad_idx = [10, 50, 200]
    data.loc[bad_idx, "temperature"] = BAD_FLAG
    bad_mask = data["temperature"] == BAD_FLAG

    result = filt_interp(data, "temperature", 0.03, SAMPLE_DT, flag=True)

    assert np.all(result[bad_mask] == BAD_FLAG), (
        "bad-flagged positions not restored after filt_interp(flag=True)"
    )


def test_filt_interp_flag_false_fills_bad_values(stage_convert):
    """With flag=False (the pipeline default), injected bad flags are interpolated away."""
    data = stage_convert.copy()
    bad_idx = [10, 50, 200]
    data.loc[bad_idx, "temperature"] = BAD_FLAG
    bad_mask = data.index.isin(bad_idx)

    result = filt_interp(data, "temperature", 0.03, SAMPLE_DT, flag=False)

    assert not np.any(result[bad_mask] == BAD_FLAG), (
        "bad-flagged values not filled when flag=False"
    )


# ---------------------------------------------------------------------------
# slope_correction
# ---------------------------------------------------------------------------


def test_slope_correction_identity_leaves_data_unchanged(stage_convert):
    """slope=1.0 must not change conductivity; salinity must equal gsw.SP_from_C of original."""
    step = {"variables": ["conductivity"], "slope": [1.0]}
    original_cond = stage_convert["conductivity"].values.copy()
    result, _ = apply_slope_correction(stage_convert.copy(), step)
    np.testing.assert_array_equal(result["conductivity"].values, original_cond)
    expected_sal = gsw.SP_from_C(
        10 * original_cond,
        t=stage_convert["temperature"].values,
        p=stage_convert["pressure"].values,
    )
    np.testing.assert_allclose(
        result["salinity"].values,
        expected_sal,
        rtol=1e-10,
        err_msg="salinity not correctly computed under identity slope",
    )


def test_slope_correction_scales_conductivity(stage_convert):
    """Conductivity must be multiplied by slope; salinity must be recomputed."""
    m = 1.001
    step = {"variables": ["conductivity"], "slope": [m]}
    original_cond = stage_convert["conductivity"].values.copy()
    result, _ = apply_slope_correction(stage_convert.copy(), step)

    np.testing.assert_allclose(
        result["conductivity"].values,
        original_cond * m,
        rtol=1e-12,
        err_msg="conductivity not scaled by slope",
    )
    expected_sal = gsw.SP_from_C(
        10 * result["conductivity"].values,
        t=result["temperature"].values,
        p=result["pressure"].values,
    )
    np.testing.assert_allclose(
        result["salinity"].values,
        expected_sal,
        rtol=1e-10,
        err_msg="salinity not recomputed after slope correction",
    )


def test_slope_correction_secondary_channel_updates_salinity2(stage_convert):
    """slope on conductivity2 must update salinity2 and leave primary conductivity unchanged."""
    m = 0.999
    step = {"variables": ["conductivity2"], "slope": [m]}
    original_cond = stage_convert["conductivity"].values.copy()
    result, _ = apply_slope_correction(stage_convert.copy(), step)

    np.testing.assert_array_equal(
        result["conductivity"].values,
        original_cond,
        err_msg="primary conductivity changed when correcting secondary channel",
    )
    assert "salinity" not in result.columns, (
        "primary salinity column created when only secondary channel was corrected"
    )
    expected_sal2 = gsw.SP_from_C(
        10 * result["conductivity2"].values,
        t=result["temperature2"].values,
        p=result["pressure"].values,
    )
    np.testing.assert_allclose(
        result["salinity2"].values,
        expected_sal2,
        rtol=1e-10,
        err_msg="salinity2 not recomputed after secondary slope correction",
    )


def test_slope_correction_text_output_format(stage_convert):
    """text output must record variable name and slope value."""
    m = 1.002
    step = {"variables": ["conductivity"], "slope": [m]}
    _, text = apply_slope_correction(stage_convert.copy(), step)
    combined = "".join(text)
    assert "slope_correction conductivity" in combined
    assert str(m) in combined


def test_slope_correction_both_channels(stage_convert):
    """Correcting both channels must create both salinity and salinity2."""
    step = {"variables": ["conductivity", "conductivity2"], "slope": [1.001, 0.999]}
    result, text = apply_slope_correction(stage_convert.copy(), step)
    assert "salinity" in result.columns
    assert "salinity2" in result.columns
    assert len([line for line in text if "slope_correction" in line]) == 4


# ---------------------------------------------------------------------------
# slope_for_correction
# ---------------------------------------------------------------------------


def test_slope_for_correction_perfect_proportional():
    """When b = m*a exactly, slope_for_correction returns m."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    m_expected = 1.005
    b = m_expected * a
    assert abs(slope_for_correction(a, b) - m_expected) < 1e-12


def test_slope_for_correction_nan_values_ignored():
    """NaN entries in either array must be excluded from the calculation."""
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([2.0, 4.0, 99.0, 8.0])
    assert abs(slope_for_correction(a, b) - 2.0) < 1e-12


def test_slope_for_correction_with_real_conductivity(stage_convert):
    """slope_for_correction on real primary vs secondary conductivity returns a plausible value."""
    a = stage_convert["conductivity"].values
    b = stage_convert["conductivity2"].values
    m = slope_for_correction(a, b)
    assert 0.9 < m < 1.1, f"implausible slope between primary and secondary conductivity: {m}"


# ---------------------------------------------------------------------------
# Up/down cast processing — alp_tau*, find_opt_alp_tat*, crosshigh, bottle_avg
#
# MIXSED2_000 has 17880 total scans; peak pressure 81.6 dbar at index 9478.
# ---------------------------------------------------------------------------

_PI, _PF = 20.0, 70.0
_TBIN = 0.1
_ALPHA_R = np.array([0.03, 0.05])
_TAU_R = np.array([5.0, 7.0])


@pytest.fixture(scope="module")
def updown_data(stage_convert):
    """Converted fixture data confirmed to contain a downcast + upcast."""
    return stage_convert.copy()


def _split_updown(data):
    """Split fixture data into down/up pressure-band arrays for alp_tau_fast."""
    pres = data["pressure"].values
    temp = data["temperature"].values
    cond = data["conductivity"].values
    peak = int(np.argmax(pres))
    mask_dn = (pres[:peak] >= _PI) & (pres[:peak] <= _PF)
    mask_up = (pres[peak + 1:] >= _PI) & (pres[peak + 1:] <= _PF)
    p_dn = pres[:peak][mask_dn]
    t_dn = temp[:peak][mask_dn]
    c_dn = cond[:peak][mask_dn]
    p_up = pres[peak + 1:][mask_up]
    t_up = temp[peak + 1:][mask_up]
    c_up = cond[peak + 1:][mask_up]
    temp_bins = np.arange(
        np.max([t_dn.min(), t_up.min()]),
        np.min([t_dn.max(), t_up.max()]),
        _TBIN,
    )
    return p_dn, t_dn, c_dn, p_up, t_up, c_up, temp_bins


def _ctd_with_scan(data):
    """Add synthetic scan numbers to a fixture DataFrame."""
    d = data.copy()
    d["scan"] = np.arange(len(d), dtype=float)
    return d


_BL = bl_reader(RAW_DIR / f"{CAST}.bl")


# alp_tau_fast


def test_alp_tau_fast_returns_scalar(updown_data):
    result = alp_tau_fast(*_split_updown(updown_data))
    assert np.isscalar(result) or isinstance(result, np.floating)


def test_alp_tau_fast_non_negative(updown_data):
    result = alp_tau_fast(*_split_updown(updown_data))
    assert result >= 0.0, f"negative error: {result}"


def test_alp_tau_fast_plausible_magnitude(updown_data):
    """Uncorrected mismatch on real data should be < 1 PSU."""
    result = alp_tau_fast(*_split_updown(updown_data))
    assert result < 1.0, f"implausibly large salinity mismatch: {result} PSU"


# find_opt_alp_tat_fast


def test_find_opt_alp_tat_fast_matrix_shape(updown_data):
    mat = find_opt_alp_tat_fast(
        _ALPHA_R, _TAU_R, updown_data,
        ["pressure", "temperature", "conductivity"],
        _PI, _PF, _TBIN, SAMPLE_DT,
    )
    assert mat.shape == (_ALPHA_R.shape[0], _TAU_R.shape[0])


def test_find_opt_alp_tat_fast_all_entries_non_negative(updown_data):
    mat = find_opt_alp_tat_fast(
        _ALPHA_R, _TAU_R, updown_data,
        ["pressure", "temperature", "conductivity"],
        _PI, _PF, _TBIN, SAMPLE_DT,
    )
    assert np.all(mat >= 0.0) or np.all(np.isnan(mat)), f"unexpected negative entry: {mat}"


def test_find_opt_alp_tat_fast_optimum_within_grid(updown_data):
    mat = find_opt_alp_tat_fast(
        _ALPHA_R, _TAU_R, updown_data,
        ["pressure", "temperature", "conductivity"],
        _PI, _PF, _TBIN, SAMPLE_DT,
    )
    r, c = np.unravel_index(np.nanargmin(mat), mat.shape)
    assert 0 <= r < len(_ALPHA_R)
    assert 0 <= c < len(_TAU_R)


# alp_tau


def test_alp_tau_returns_scalar(updown_data):
    result = alp_tau(
        updown_data.copy(),
        ["pressure", "temperature", "conductivity"],
        _PI, _PF, _TBIN,
    )
    assert isinstance(result, float), f"expected float, got {type(result)}"


def test_alp_tau_non_negative(updown_data):
    result = alp_tau(
        updown_data.copy(),
        ["pressure", "temperature", "conductivity"],
        _PI, _PF, _TBIN,
    )
    assert result >= 0.0


def test_alp_tau_does_not_mutate_input(updown_data):
    """alp_tau overwrites param[2] internally — must not touch the caller's DataFrame."""
    cond_before = updown_data["conductivity"].values.copy()
    alp_tau(
        updown_data.copy(),
        ["pressure", "temperature", "conductivity"],
        _PI, _PF, _TBIN,
    )
    np.testing.assert_array_equal(
        updown_data["conductivity"].values, cond_before,
        err_msg="alp_tau mutated the caller's conductivity column",
    )


# find_opt_alp_tat


def test_find_opt_alp_tat_matrix_shape(updown_data):
    data = updown_data.copy()
    data["conductivity_corr"] = data["conductivity"].copy()
    mat = find_opt_alp_tat(
        _ALPHA_R, _TAU_R, data,
        ["pressure", "temperature", "conductivity", "conductivity_corr"],
        _PI, _PF, _TBIN, figure=False,
    )
    assert mat.shape == (_ALPHA_R.shape[0], _TAU_R.shape[0])


def test_find_opt_alp_tat_all_entries_non_negative(updown_data):
    data = updown_data.copy()
    data["conductivity_corr"] = data["conductivity"].copy()
    mat = find_opt_alp_tat(
        _ALPHA_R, _TAU_R, data,
        ["pressure", "temperature", "conductivity", "conductivity_corr"],
        _PI, _PF, _TBIN, figure=False,
    )
    assert np.all(mat >= 0.0), f"negative error: {mat}"


def test_find_opt_alp_tat_consistent_with_fast_variant(updown_data):
    """Both variants should agree on which (alpha, tau) minimises the error."""
    import matplotlib.pyplot as plt

    data = updown_data.copy()
    data["conductivity_corr"] = data["conductivity"].copy()
    slow = find_opt_alp_tat(
        _ALPHA_R, _TAU_R, data,
        ["pressure", "temperature", "conductivity", "conductivity_corr"],
        _PI, _PF, _TBIN, figure=False,
    )
    fast = find_opt_alp_tat_fast(
        _ALPHA_R, _TAU_R, updown_data,
        ["pressure", "temperature", "conductivity"],
        _PI, _PF, _TBIN, SAMPLE_DT,
    )
    plt.close("all")
    assert np.unravel_index(np.nanargmin(slow), slow.shape) == np.unravel_index(
        np.nanargmin(fast), fast.shape
    ), f"slow and fast variants disagree on optimal (alpha, tau): slow={slow}, fast={fast}"


# crosshigh


def test_crosshigh_returns_four_values(updown_data):
    import matplotlib.pyplot as plt

    result = crosshigh(
        updown_data,
        ["pressure", "temperature", "conductivity"],
        maxL=5, pi=_PI, pf=_PF, si=SAMPLE_DT, high=False,
    )
    plt.close("all")
    assert len(result) == 4, f"expected 4 values, got {len(result)}"


def test_crosshigh_correlations_in_valid_range(updown_data):
    import matplotlib.pyplot as plt

    result = crosshigh(
        updown_data,
        ["pressure", "temperature", "conductivity"],
        maxL=5, pi=_PI, pf=_PF, si=SAMPLE_DT, high=False,
    )
    plt.close("all")
    corr_up, _, corr_dn, _ = result
    assert -1.0 <= corr_up <= 1.0, f"corr_up out of range: {corr_up}"
    assert -1.0 <= corr_dn <= 1.0, f"corr_dn out of range: {corr_dn}"


def test_crosshigh_lag_bounded_by_maxL(updown_data):
    import matplotlib.pyplot as plt

    maxL = 5
    result = crosshigh(
        updown_data,
        ["pressure", "temperature", "conductivity"],
        maxL=maxL, pi=_PI, pf=_PF, si=SAMPLE_DT, high=False,
    )
    plt.close("all")
    _, lag_up, _, lag_dn = result
    limit = maxL * SAMPLE_DT
    assert abs(lag_up) <= limit + 1e-9, f"lag_up={lag_up} exceeds ±{limit}"
    assert abs(lag_dn) <= limit + 1e-9, f"lag_dn={lag_dn} exceeds ±{limit}"


# bottle_avg


def test_bottle_avg_returns_dataframe(updown_data):
    result = bottle_avg(_ctd_with_scan(updown_data), _BL, scan_offset=5, scan_duration=10, samp_int=24)
    assert isinstance(result, pd.DataFrame)


def test_bottle_avg_row_count_matches_bottles(updown_data):
    """One row per bottle fire in the .bl file."""
    result = bottle_avg(_ctd_with_scan(updown_data), _BL, scan_offset=5, scan_duration=10, samp_int=24)
    assert len(result) == len(_BL)


def test_bottle_avg_required_columns_present(updown_data):
    result = bottle_avg(_ctd_with_scan(updown_data), _BL, scan_offset=5, scan_duration=10, samp_int=24)
    expected = {"bottle", "conductivity", "conductivity2",
                "temperature", "temperature2", "latitude", "longitude", "pressure"}
    assert expected.issubset(set(result.columns))


def test_bottle_avg_pressure_within_cast_range(updown_data):
    """Averaged pressure must lie within the fixture cast's pressure range."""
    ctd = _ctd_with_scan(updown_data)
    p_min = float(updown_data["pressure"].min())
    p_max = float(updown_data["pressure"].max())
    result = bottle_avg(ctd, _BL, scan_offset=5, scan_duration=10, samp_int=24)
    for p in result["pressure"]:
        assert p_min <= p <= p_max, f"averaged pressure {p} outside cast range [{p_min}, {p_max}]"
