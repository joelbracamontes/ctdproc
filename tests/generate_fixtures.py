"""Generate reference CSV fixtures for pipeline regression tests.

Run this once (from the repo root) to create tests/fixtures/reference/*.csv:

    python tests/generate_fixtures.py

Commit the resulting CSVs.  Re-run only when a deliberate change to the
pipeline output is intended — the regression tests compare against these files.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ctdproc.load import convert_data, load_raw_data
from ctdproc.main import CTD_process
from ctdproc.pipeline import (
    apply_align,
    apply_bin,
    apply_celltm,
    apply_loop_edit,
    apply_low_filter,
    apply_wild_edit,
)

FIXTURES_DIR = ROOT / "tests" / "fixtures"
RAW_DIR = FIXTURES_DIR / "raw"
REF_DIR = FIXTURES_DIR / "reference"
REF_DIR.mkdir(exist_ok=True)

CAST = "MIXSED2_000"
SAMPLE_DT = 1 / 24
BAD_FLAG = np.float64(-9.99e-29)

with open(FIXTURES_DIR / "config.yaml") as _f:
    CONFIG_FEATURES = yaml.safe_load(_f)["features"]


def save(df: pd.DataFrame, name: str) -> None:
    path = REF_DIR / f"{name}.csv.gz"
    df.to_csv(path, index=False, float_format="%.6g", compression="gzip")
    print(
        f"  saved {path.name}  ({path.stat().st_size // 1024}KB, {len(df)} rows × {len(df.columns)} cols)"
    )


if __name__ == "__main__":
    steps = {s["name"]: s for s in CONFIG_FEATURES}

    print(f"Loading {CAST} ...")
    raw = load_raw_data(RAW_DIR / f"{CAST}.hex")
    data = convert_data(raw, RAW_DIR / f"{CAST}.xmlcon", SAMPLE_DT)
    save(data, "stage_00_convert")

    print("align ...")
    data, _ = apply_align(data, steps["align"], SAMPLE_DT, BAD_FLAG)
    save(data, "stage_01_align")

    print("wild_edit ...")
    data, _ = apply_wild_edit(data, steps["wild_edit"], BAD_FLAG)
    save(data, "stage_02_wild_edit")

    print("low_filter ...")
    data, _ = apply_low_filter(data, steps["low_filter"], SAMPLE_DT)
    save(data, "stage_03_low_filter")

    print("celltm ...")
    data, _ = apply_celltm(data, steps["celltm"], SAMPLE_DT, BAD_FLAG)
    save(data, "stage_04_celltm")

    print("loop_edit ...")
    data, _ = apply_loop_edit(data, steps["loop_edit"], SAMPLE_DT, BAD_FLAG)
    save(data, "stage_05_loop_edit")

    print("bin ...")
    bin_results, _, output_names = apply_bin(data, steps["bin"], BAD_FLAG, CAST)
    # output_names: ['MIXSED2_000_24scans.cnv', 'downMIXSED2_000_1dbar.cnv']
    for bin_df, name in zip(bin_results, output_names):
        label = Path(name).stem  # e.g. MIXSED2_000_24scans
        save(bin_df, f"stage_06_bin_{label}")

    print("Generating .cnv reference files via full pipeline ...")
    cfg_text = (FIXTURES_DIR / "config.yaml").read_text()
    cfg_text = cfg_text.replace("__RAW_DIR__", RAW_DIR.resolve().as_posix())
    cfg_text = cfg_text.replace("__OUT_DIR__", REF_DIR.resolve().as_posix())
    tmp_cfg = REF_DIR / "_tmp_config.yaml"
    tmp_cfg.write_text(cfg_text)
    try:
        CTD_process(str(tmp_cfg))
    finally:
        tmp_cfg.unlink()

    print("Done.")
