# ctdproc-MIXSED2

Processes shipboard Sea-Bird SBE 911plus CTD data: raw `.hex`/`.XMLCON` in,
calibration-corrected `.cnv` out. Built for the **MIXSED2** cruise. Feeds
its output into [`caldip-MIXSED2`](../caldip-MIXSED2) for calibration-dip
statistics.

---

## Layout

```text
ctdproc-MIXSED2_autoconfig/
├── src/
│   ├── ctdproc/                # Pipeline: main.py + 3 finders (below)
│   └── seabirdscientific/      # Vendored Sea-Bird toolkit, v3.1.2
├── tests/                      # pytest regression suite
├── pyproject.toml / requirements.txt / environment.yml
└── README.md
```

`src/ctdproc/load.py` auto-detects sensor config from the companion
`.XMLCON` (no hardcoded sensor list), so it reads any cruise's
`.hex`/`.XMLCON` pair unchanged. `src/seabirdscientific/` is vendored (not
pip-installed) so it can carry one local patch (latin-1 decoding) on top
of upstream 3.1.2.

---

## Install

```bash
pip install -r requirements.txt && pip install -e .
# or: conda env create -f environment.yml && conda activate ctdproc && pip install -e .
```

Then `pytest tests/` to check it works.

---

## Running the pipeline

```text
data/ (raw .hex/.XMLCON)  +  proc/config.yaml  →  python -m ctdproc config.yaml  (run from proc/)
```

```yaml
constants:
  sample_interval: 1/24
  name: "MIXSED2"
  ind: [30, 38]          # cast range -> name_030.hex ... name_038.hex
  convert: 'yes'
  time_ref: 'System UTC'

features:                 # only uncommented steps run, in this order
  # - name: "align"        {variables, offset}            # from ctdalign_finder
  # - name: "wild_edit"    {variables, std_pass_1, std_pass_2, scans_per_block, distance_to_mean}
  # - name: "low_filter"   {variables, time_constant}
  # - name: "celltm"       {variables, amplitude, time_constant}  # from cellTM_finder
  # - name: "slope_corr"   {variables, slope}             # from slopecorrection_finder
  # - name: "loop_edit"    # NOT for caldip casts - flags bottle-stops as ship-roll noise
  # - name: "bin"          {bin_type, bin_size, type_profile}
```

- `slope_corr` must come after `celltm` (it recomputes salinity from
  whatever conductivity it's handed).
- `.cnv` output currently only carries P/T/C/salinity.

---

## Diagnostic finders

Each fits coefficients for one `features:` step above. Run from a `proc/`
dir, in this order: **align → celltm → slope**.

| Finder | Fits | Config key | Run | Output |
|---|---|---|---|---|
| `ctdalign_finder` | sensor time-lag | `alignment_settings` | `python -m ctdproc.ctdalign_finder config_align.yaml` | `alignment_report.csv` — paste straight into `align`'s `offset:` |
| `cellTM_finder` | thermal-mass α/τ | `celltm_settings` | `python -m ctdproc.cellTM_finder config_celltm.yaml` | `alpha_tau_report.csv` + contour plots. **Check `edge_pinned`** — a fit pinned at the search grid's edge (α 0.01–0.09, τ 1–16) means the real optimum is outside it |
| `slopecorrection_finder` | conductivity slope | `slope_settings` | `python -m ctdproc.slopecorrection_finder config_slopecorr.yaml` | `slope_report.csv` + ratio plots. Iterative 3σ outlier rejection |

Example `slope_settings`:
```yaml
slope_settings:
  bottle_sal_file: "Bottle_Salinity.csv"   # columns: Station, No_bottle, Sal_avg
  scan_offset: [2, 2]        # seconds before bottle closure
  scan_duration: [2, 2]
  temperature_sensors: ["temperature", "temperature2"]
  conductivity_sensors: ["conductivity", "conductivity2"]
```
(`[2, 2]` matches the GO-SHIP standard, confirmed against this cruise's
own Portasal/Autosal report.)

---

## Testing

```bash
pytest tests/
```

`test_pipeline_regression.py` diffs a real cast's output, stage by stage,
against `tests/fixtures/reference/`. Only regenerate those with
`python tests/generate_fixtures.py` when a behavior change is intentional.

---

## Known gaps

- `.cnv` output skips oxygen/chlorophyll/turbidity/PAR (needs per-sensor
  naming from the XMLCON, not yet built).
- `ctdalign_finder` has no sub-sample precision (whole-scan lags only).
- `cellTM_finder`'s search grid (α 0.01–0.09, τ 1–16) is fixed, not
  configurable per cast/campaign.
- Vendored `seabirdscientific` just went 2.7.8 → 3.1.2 - a few breaking
  changes were found and fixed along the way (return-type changes,
  a removed `scalar` param, a coefficient casing change, `loop_edit`'s
  return semantics), confirmed via the regression suite.

---

## Disclaimers

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
