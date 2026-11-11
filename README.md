# ctdproc-MIXSED2

A pipeline for processing shipboard Sea-Bird SBE 911plus CTD data: raw
`.hex`/`.XMLCON` in, calibration-corrected `.cnv` out. Built for the
**MIXSED project** cruises (2012, 2017, 2018, 2026_1/MIXSED1, 2026_2/MIXSED2),
and shares its output with the separate
[`caldip-MIXSED2`](../caldip-MIXSED2) package, which does the calibration-dip
(CTD-vs-instrument) comparison statistics.

> **Status: campaign-specific tooling, not a general-purpose release.**
> This is a working cruise-processing codebase, not a polished library. It
> has hardcoded assumptions about the MIXSED campaigns baked into it (see
> **Disclaimers** below) and has not been merged into the main `ctdam`
> package it's meant to eventually feed into.

---

## What's in here

```text
ctdproc-MIXSED2_autoconfig/
├── data/                       # Raw MIXSED2 cruise data (.hex/.hdr/.bl/.XMLCON)
├── src/
│   ├── ctdproc/                # The actual pipeline - see below
│   └── seabirdscientific/      # Vendored Sea-Bird Scientific Community Toolkit
├── tools/
│   ├── caldip/                 # Bridges ctdproc output -> caldip-MIXSED2's input
│   └── salinometer/            # Bottle-salinity (.DAT -> .COR) processing for slope_corr
├── tests/                      # Regression tests (pytest)
├── 2012/ 2017/ 2018/ 2026_1/ 2026_2/   # One self-contained data+proc folder per cruise/year
├── notebooks/                   # Exploratory Jupyter notebooks
├── FINDINGS_2026-09-29.md       # Running dev log: bugs found, decisions made, why
├── pyproject.toml / requirements.txt / environment.yml
└── README.md
```

* **`src/ctdproc/`** is where the actual processing logic lives:
  `main.py` (the pipeline entry point), `load.py` (reading raw data - see
  disclaimer below, this one is hardcoded to MIXSED2's instrument layout),
  `pipeline.py` (each processing step: align/wild_edit/celltm/slope_corr/
  loop_edit/bin), `parsers.py` (XMLCON parsing), `processing.py` (the
  underlying math: cross-correlation, cell-thermal-mass grid search, slope
  fitting), and the three **finder** scripts covered in detail below.
* **`src/seabirdscientific/`** is a vendored (not pip-installed) copy of
  Sea-Bird's own processing toolkit, currently **v3.1.2** (upgraded from an
  older v2.7.8 fork in this session - see **Disclaimers**), with one local
  patch re-applied on top (a latin-1 encoding fix for reading `.cnv`/config
  files with non-UTF-8 header bytes).

---

## Installation & Setup

### Option 1: pip
```bash
pip install -r requirements.txt
pip install -e .
```

### Option 2: Conda/Mamba
```bash
conda env create -f environment.yml
conda activate ctdproc
pip install -e .
```

Both pull in `xarray` and `seawater` in addition to the core Sea-Bird/scipy
stack - these became required when the vendored `seabirdscientific` was
upgraded to 3.1.2 (its `read_hex_file`/`bin_average` now work on
`xr.Dataset`, and the deprecated EOS-80 shims use `seawater`). If you're
working from an older environment that predates that upgrade, you'll need
to add these two manually.

To run the test suite, also install `pytest` (included in
`requirements.txt`, not in `pyproject.toml`'s runtime dependencies, since
it's dev-only):
```bash
pytest tests/
```

---

## How to Run the Main Pipeline

Create a `proc/` working directory alongside `data/`:

```text
ctdproc-MIXSED2/
├── data/               # Raw .hex/.XMLCON files
├── proc/               # Your execution directory
│   └── config.yaml     # Campaign configuration file
├── src/
└── ...
```

```bash
cd proc
python -m ctdproc config.yaml
```

Each of the `2012/`, `2017/`, `2018/`, `2026_1/`, `2026_2/` folders is a
real, working example of this layout - look at `<year>/proc/config.yaml`
and `<year>/README.md` for a concrete, already-tuned configuration.

### `config.yaml`

```yaml
constants:
  sample_interval: 1/24
  name: "MIXSED2"
  ind: [30, 38]          # Cast index range: loops over name_030.hex ... name_038.hex
  # sind: [30, 32, 35]    # Alternative to `ind` - an explicit list instead of a range.
                          # Set only one of ind/sind, not both.
  convert: 'yes'          # 'yes' -> also write a nominal 24 Hz .cnv per cast
  time_ref: 'System UTC'  # Which raw-header timestamp line to anchor elapsed_time to
  # data_dir: '../data'   # Optional, default shown
  # output_dir: '.'       # Optional, default shown

features:
  # Only the uncommented steps below run, in the order listed.
  # - name: "align"
  #   variables: ["conductivity", "conductivity2"]
  #   offset: [0.073, 0.073]              # from ctdalign_finder's alignment_report.csv
  # - name: "wild_edit"
  #   variables: ["temperature", "temperature2", "conductivity", "conductivity2", "pressure"]
  #   std_pass_1: [2.0, 2.0, 2.0, 2.0, 2.0]
  #   std_pass_2: [5.0, 5.0, 5.0, 5.0, 5.0]
  #   scans_per_block: [100, 100, 100, 100, 100]
  #   distance_to_mean: [0.0, 0.0, 0.0, 0.0, 0.0]
  # - name: "low_filter"
  #   variables: ["pressure", "temperature", "conductivity"]
  #   time_constant: [0.03, 0.03, 0.03]
  # - name: "celltm"
  #   variables: ["conductivity", "conductivity2"]
  #   amplitude: [0.03, 0.03]             # from cellTM_finder's alpha_tau_report.csv
  #   time_constant: [7.0, 7.0]
  # - name: "slope_corr"
  #   variables: ["conductivity", "conductivity2"]
  #   slope: [1.00003, 0.99997]           # from slopecorrection_finder's slope_report.csv
  # - name: "loop_edit"                    # NOT suitable for calibration-dip casts - see below
  #   min_velocity: [0.25]
  #   window_size: [3]
  #   mean_speed_percent: [20]
  #   remove_surface_soak: [True]
  #   min_soak_depth: [5]
  #   max_soak_depth: [20]
  #   use_deck_pressure_offset: [False]
  # - name: "bin"
  #   bin_type: ["scans", "pressure"]
  #   bin_size: [24.0, 1.0]
  #   type_profile: ["BOTH", "DOWN"]
```

Notes:
- **Step order matters.** `slope_corr` should come after `celltm` (it
  recomputes salinity from whatever conductivity it's handed, so anything
  touching conductivity afterward is silently not reflected in salinity).
- **`loop_edit` is not suitable for calibration-dip (caldip) casts.** It
  flags ship-roll pressure reversals as bad data - but a caldip cast has
  deliberate multi-minute stationary bottle-stops that look exactly like
  that to the algorithm. The real caldip configs (see `tools/caldip/`)
  deliberately omit both `loop_edit` and `bin`.
- **CNV output currently only carries P/T/C/salinity.** Other sensors
  (oxygen, chlorophyll, turbidity, PAR, ...) are deliberately excluded from
  `main.py`'s `.cnv` writer for now - see **Under Development** below.

---

## Diagnostic Tools ("Finders")

These three scripts don't produce final data products - they fit the
*coefficients* that go into a real `config.yaml`'s `features:` list above.
Run each from a `proc/`-style directory (next to the relevant `data/`), each
on its own YAML config, in roughly this order: **align → celltm → slope**.

### 1. `ctdalign_finder` — sensor time-lag

Finds the time lag between a lead/lag sensor pair (e.g. pressure vs.
conductivity) via cross-correlation, so you can line up a lagging sensor's
response in time.

**Config (`config_align.yaml`):**
```yaml
constants:
  sample_interval: 1/24
  name: "MIXSED2"
  ind: [30, 38]

alignment_settings:
  variable_lead: ["temperature"]
  variable_lag: ["conductivity"]
  max_lag: [15]
  pressure_upper: [30.0]
  pressure_lower: [500.0]
```

**Run:**
```bash
python -m ctdproc.ctdalign_finder config_align.yaml
```

**Output: `alignment_report.csv`** — per-cast `lag_down_cast`/`lag_up_cast`,
plus a campaign-average row. These values are already sign-corrected to
paste directly into an `align` step's `offset:` list.

### 2. `cellTM_finder` — cell thermal mass (alpha, tau)

Grid-searches the conductivity cell thermal-mass correction (`amplitude`
α, `time_constant` τ) that minimizes the down-cast/up-cast salinity
mismatch, per cast.

**Config (`config_celltm.yaml`):**
```yaml
constants:
  sample_interval: 1/24
  name: "MIXSED2"
  ind: [30, 38]

celltm_settings:
  pressure_upper: [30.0]
  pressure_lower: [500.0]
  bin_temperature: [0.5]
  temperature_touse: ["temperature"]
  conductivity_touse: ["conductivity"]
```

**Run:**
```bash
python -m ctdproc.cellTM_finder config_celltm.yaml
```

**Output: `alpha_tau_report.csv`** (per-cast `tau`/`alpha` + a campaign
average) and one `celltm_plot_cast_<NNN>_<sensor>.png` contour plot per
cast/sensor. **Read the console output and the `edge_pinned` column before
trusting the average** — a fit that lands on (or within 2 grid points of)
the search grid's edge (`alpha` in `[0.01, 0.09]`, `tau` in `[1, 16]`)
means the true optimum is likely outside that range, not actually found.
Edge-pinned casts are excluded from the campaign average automatically,
but if most casts for a sensor are edge-pinned, the search grid itself
needs widening (seen on 2017 — not yet done, see **Under Development**).

### 3. `slopecorrection_finder` — conductivity slope (gain)

Fits a conductivity slope correction per sensor pair from CTD-vs-bottle
(salinometer) comparisons at each bottle stop, with iterative 3σ outlier
rejection.

**Config (`config_slopecorr.yaml`):**
```yaml
constants:
  sample_interval: 1/24
  name: "MIXSED2"

features:
  # same align/low_filter/wild_edit/celltm steps as your production config -
  # this re-runs stations through them before comparing to bottle values
  - name: "celltm"
    variables: ["conductivity", "conductivity2"]
    amplitude: [0.0385, 0.020]
    time_constant: [7.190, 3.714]

slope_settings:
  bottle_sal_file: "Bottle_Salinity.csv"   # columns: Station, No_bottle, Sal_avg
  scan_offset: [2, 2]        # seconds from bottle closure to start averaging
  scan_duration: [2, 2]      # seconds to average over
  temperature_sensors: ["temperature", "temperature2"]
  conductivity_sensors: ["conductivity", "conductivity2"]
```

**Run:**
```bash
python -m ctdproc.slopecorrection_finder config_slopecorr.yaml
```

**Output: `slope_report.csv`** (summary `slope_coefficient` per sensor
pair + `n_outliers_excluded`, plus a full per-bottle detail log) and two
`Conductivity_ratio_report_<sensor>_{all,clean}.png` diagnostic plots per
sensor pair. The `scan_offset`/`scan_duration` of `[2, 2]` (2 s, ending at
bottle closure) matches the GO-SHIP standard — confirmed against this
cruise's own Portasal/Autosal report; see `FINDINGS_2026-09-29.md` section
5 for the full derivation if you need to justify a different window.

---

## `tools/`

- **`tools/caldip/`** — bridges this package's output into the separate
  `caldip-MIXSED2` repository: `run_caldip_processing.py` runs each
  year's calibration-dip casts through the main pipeline
  (align/wild_edit/low_filter/celltm/slope_corr only — `loop_edit`/`bin`
  omitted, per the note above) with that year's confirmed coefficients;
  `ctdproc_to_cnv.py` then translates the resulting `.cnv` into the exact
  real Sea-Bird `.cnv` format `caldip`'s own reader expects (this is a real
  format-compatibility shim, not cosmetic — see the file's own docstring);
  `plot_ts_stages.py` plots a cast's T-S diagram at each processing stage.
- **`tools/salinometer/`** — converts raw salinometer `.DAT` sessions into
  drift-corrected `.COR` bottle salinities (and cross-checks that
  conversion several different ways: `validate_dat_to_cor*.py`,
  `compare_rates.py`, `diagnose_residual.py`) — these are what
  `Bottle_Salinity.csv` (the `slopecorrection_finder` input) is built from.
  Two of the `build_bottle_salinity_*.py` scripts are year-specific one-off
  builders, not generic tools.

---

## Testing

```bash
pytest tests/
```

`test_pipeline_regression.py` is the main safety net: it runs a real cast
through each pipeline stage and the full pipeline, and diffs the output
against committed fixtures in `tests/fixtures/reference/`. If you
intentionally change processing behavior, regenerate the fixtures with
`python tests/generate_fixtures.py` — **only do this when the behavior
change is intentional**, since it will silently paper over a real
regression otherwise. `test_eos80.py` and `test_conversion_vectorize.py`
check the vendored `seabirdscientific` module directly.

---

## Under Development / Known Gaps

- **Oxygen/chlorophyll/turbidity/PAR are not written to the `.cnv`
  output.** `main.py` only emits P/T/C/salinity — caldip doesn't need the
  others yet, and writing them properly needs per-sensor-model naming
  pulled from the XMLCON (the next item), not the one fixed SBE name each
  P/T/C/S currently gets.
- **`load_xmlcon` doesn't parse `TurbidityMeter` coefficients at all** —
  a deliberately left gap (confirmed, not accidental).
- **No sub-sample alignment precision** in `ctdalign_finder` — lags are
  found to the nearest whole scan; a genuinely sub-scan lag can't be
  resolved with the current method.
- **`cellTM_finder`'s search grid is fixed** (`alpha` 0.01–0.09, `tau`
  1–16) and not configurable per campaign. At least one year (2017) shows
  signs the true optimum sits outside this range (alpha values clustering
  at the edge) — grid was never widened/re-run to check.
- **None of this has been merged upstream** into either the original
  `ctdproc-MIXSED2` project or the eventual target, `ctdam` — everything
  here is local to this copy (hence the `_autoconfig` suffix).
- The vendored `seabirdscientific` was upgraded from a v2.7.8 fork to
  upstream v3.1.2 in a recent session; this surfaced and fixed several
  breaking changes (return-type changes in `read_hex_file`/`bin_average`,
  a removed `scalar` param on `convert_conductivity`, a coefficient
  casing change, and `loop_edit`'s return semantics). The regression suite
  passes end-to-end, but this vendored copy has now diverged from both the
  old fork *and* from whatever `seabirdscientific` version the sibling
  `ctdam` package's pip dependency uses — reconciling the two was
  explicitly deferred, not attempted.

See `FINDINGS_2026-09-29.md` for the full, dated history of what was found
and fixed, including the reasoning behind each — it's long, but it's the
real decision log for this codebase, not just a changelog.

---

## Disclaimers

- **This is campaign-specific, not generic.** `src/ctdproc/load.py` in
  particular is hardcoded to MIXSED2's instrument/sensor layout, not a
  general SBE 911plus reader. Pointing this pipeline at a different
  cruise's raw data without checking that file first will likely break
  silently or loudly, not gracefully.
- **The vendored `seabirdscientific/` has local patches.** It is not a
  clean copy of any released version — don't `pip install
  seabirdscientific` over it or copy files from a fresh download without
  diffing first; you will lose the re-applied fix (a latin-1 decode fix in
  `instrument_data.py`) and risk reintroducing the breaking changes listed
  above.
- **Config-driven, not validated.** None of the YAML configs are
  schema-checked — a misspelled key, wrong list length, or mismatched
  `variables:`/coefficient-list pairing in `features:` will generally fail
  with a raw Python traceback (e.g. an `IndexError` from `zip`-ing
  mismatched-length lists) rather than a clear error message.
- **`data/` contains real, unrepeatable cruise data.** Nothing in this
  repo regenerates it — treat it as read-only input, not a working file to
  edit or an output path to point tools at by accident.
- **Outlier rejection and edge-pinning exclusions are automatic but
  silent unless you read the console output / CSV columns.** A finder
  script will happily report a campaign average computed from, say, 3 out
  of 10 casts if the other 7 were excluded — the exclusion counts are in
  the output, but nothing stops you from only looking at the averages row.
