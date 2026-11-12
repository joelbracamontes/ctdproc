"""Tests for seabirdscientific.eos80_processing.

Uses real fixture data (MIXSED2_000 cast) for T and P; salinity is
computed from fixture conductivity via gsw.SP_from_C.
"""

import sys
from pathlib import Path

import gsw
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ctdproc.load import convert_data, load_raw_data
from seabirdscientific import eos80_processing as eos80

_RAW_DIR = ROOT / "tests" / "fixtures" / "raw"
SAMPLE_INTERVAL = 1 / 24

_raw = load_raw_data(_RAW_DIR / "MIXSED2_000.hex")
_data = convert_data(_raw, _RAW_DIR / "MIXSED2_000.xmlcon", SAMPLE_INTERVAL)

TEMP = _data["temperature"].to_numpy(dtype=float)
PRES = _data["pressure"].to_numpy(dtype=float)
SAL = gsw.SP_from_C(10 * _data["conductivity"].to_numpy(dtype=float), t=TEMP, p=PRES)

# Use a mid-cast window well below the surface soak
_WINDOW = slice(500, 600)
TEMP_WIN = TEMP[_WINDOW]
PRES_WIN = PRES[_WINDOW]
SAL_WIN = SAL[_WINDOW]

GRAVITY = 9.81


# ---------------------------------------------------------------------------
# adiabatic_temperature_gradient
# ---------------------------------------------------------------------------


def test_atg_returns_array_same_shape():
    atg = eos80.adiabatic_temperature_gradient(SAL_WIN, TEMP_WIN, PRES_WIN)
    assert atg.shape == TEMP_WIN.shape


def test_atg_physically_plausible_magnitude():
    """ATG for seawater is typically 0.1–0.4 m°C/dbar."""
    atg = eos80.adiabatic_temperature_gradient(SAL_WIN, TEMP_WIN, PRES_WIN)
    assert np.all(atg > 0.0), "ATG should be positive"
    assert np.all(atg < 1e-3), "ATG should be < 1 m°C/dbar"


def test_atg_scalar_inputs():
    """Should broadcast correctly with scalar inputs."""
    val = eos80.adiabatic_temperature_gradient(
        np.array([35.0]), np.array([10.0]), np.array([100.0])
    )
    assert val.shape == (1,)
    assert 1e-4 < val[0] < 5e-4


# ---------------------------------------------------------------------------
# density
# ---------------------------------------------------------------------------


def test_density_returns_array_same_shape():
    rho = eos80.density(SAL_WIN, TEMP_WIN, PRES_WIN)
    assert rho.shape == TEMP_WIN.shape


def test_density_physically_plausible_range():
    """density() returns sigma (full density - 1000), per SBE Data Processing's
    own convention - seawater sigma should be 20-35 kg/m3 in typical conditions."""
    rho = eos80.density(SAL_WIN, TEMP_WIN, PRES_WIN)
    assert np.all(rho > 20), "density below plausible range"
    assert np.all(rho < 60), "density above plausible range"


def test_density_increases_with_pressure():
    """Density should increase with pressure at constant T and S."""
    t = np.array([10.0, 10.0])
    s = np.array([35.0, 35.0])
    p = np.array([0.0, 1000.0])
    rho = eos80.density(s, t, p)
    assert rho[1] > rho[0]


# ---------------------------------------------------------------------------
# potential_temperature
# ---------------------------------------------------------------------------


def test_potential_temperature_returns_array_same_shape():
    pr = np.zeros_like(PRES_WIN)
    theta = eos80.potential_temperature(SAL_WIN, TEMP_WIN, PRES_WIN, pr)
    assert theta.shape == TEMP_WIN.shape


def test_potential_temperature_near_surface_close_to_insitu():
    """At low pressure, potential and in-situ temperature should be nearly equal."""
    p_surface = np.array([5.0, 10.0, 15.0])
    t_surface = np.array([12.0, 11.5, 11.0])
    s_surface = np.array([35.0, 35.0, 35.0])
    pr = np.zeros(3)
    theta = eos80.potential_temperature(s_surface, t_surface, p_surface, pr)
    np.testing.assert_allclose(
        theta, t_surface, atol=0.01,
        err_msg="potential temp should be ~= in-situ near surface",
    )


def test_potential_temperature_slightly_cooler_at_depth():
    """Potential temperature should be cooler than in-situ temperature at depth."""
    p_deep = np.array([1000.0])
    t_deep = np.array([4.0])
    s_deep = np.array([35.0])
    pr = np.zeros(1)
    theta = eos80.potential_temperature(s_deep, t_deep, p_deep, pr)
    assert theta[0] < t_deep[0], "potential temp should be < in-situ at depth"


# ---------------------------------------------------------------------------
# bouyancy_frequency
# ---------------------------------------------------------------------------


def test_bouyancy_frequency_returns_scalar():
    n2 = eos80.bouyancy_frequency(TEMP_WIN, SAL_WIN, PRES_WIN, GRAVITY)
    assert np.isscalar(n2) or n2.shape == ()


def test_bouyancy_frequency_positive_in_stable_water():
    """N² should be positive in a stably stratified water column."""
    n2 = eos80.bouyancy_frequency(TEMP_WIN, SAL_WIN, PRES_WIN, GRAVITY)
    assert n2 > 0, f"expected positive N² in stable water column, got {n2}"


def test_bouyancy_frequency_physically_plausible_magnitude():
    """N² for open ocean is typically 1e-7 to 1e-4 s⁻²."""
    n2 = eos80.bouyancy_frequency(TEMP_WIN, SAL_WIN, PRES_WIN, GRAVITY)
    assert 0 < n2 < 1e-2, f"N² = {n2} is outside plausible range"
