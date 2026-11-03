# Anomaly stream simulator

A seeded, configurable, synthetic multi-channel data stream simulator. It injects known anomalies and records exact ground truth. It is the data foundation for a small LLM that answers questions about abnormalities in a sensor stream (inspired by detector and target monitoring for a nuclear physics experiment).

Everything is synthetic. No real experiment data, no network calls. Channel names are generic placeholders.

This README holds progress notes. The full reference (every config field, every anomaly type, label schema, replay contract) is planned for Phase 4.

## Status

| Phase | What | State |
|---|---|---|
| 1 | Core simulator: channels, 10 anomaly types, labels, config validation, replay, CSV/JSON output | Done, committed |
| 2 | Batch generation: randomized scenarios, train/val/test split, manifest, summary table | Done, awaiting approval |
| 3 | Inspection plots | Not started |
| 4 | Full README | Not started |
| 5 | Optional: reference detectors and scoring (only on request) | Not started |

Tests: 103 passing (`pytest`).

## Setup

Python 3.11+ is required. The project was built with Python 3.14.2.

    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m pytest

All dependencies are pinned in `requirements.txt` (numpy, pandas, pyyaml, matplotlib, pytest and their dependencies).

## Usage

One run from a config:

    python scripts/make_run.py configs/mixed_demo.yaml

Writes `out/mixed_demo/data.csv`, `labels.json`, `config_used.yaml`.

A batch of randomized runs:

    python scripts/make_batch.py --n 200 --master-seed 1

Writes `out/batch/<split>/run_XXXX/` (same three files) and `out/batch/manifest.json`, then prints a table of events per anomaly type per split. Use `--help` for all options. A 200-run batch takes about 30 s and about 250 MB.

Replay a run as a stream (labels are never exposed):

    from sim import replay
    for timestamp, values in replay("configs/mixed_demo.yaml"):
        ...  # values is {channel: value}; a channel appears only when it has a sample

## Layout

    configs/   healthy.yaml, mixed_demo.yaml
    sim/       scenario (config), channels (healthy model), anomalies, labels, stream (generate and replay), io, batch
    scripts/   make_run.py, make_batch.py
    tests/     pytest suite
    out/       generated data (gitignored)

## Design decisions so far

Simulator (Phase 1):
- Same seed and config give byte-identical output. Each channel draws noise from its own child of `SeedSequence(seed)`, so adding or removing an anomaly never changes the healthy noise.
- `sigma` is a channel's total healthy standard deviation. For a coupled channel its own noise is `sqrt(sigma^2 - (gain * sigma_source)^2)`, and a gain that leaves no own noise is rejected. Default coupling gain is 0.15.
- A coupled channel follows its source's total deviation, including injected anomalies. The exception is measurement faults (`dropout`, `stuck_sensor`), which are not seen by coupled channels.
- Every anomaly with a magnitude takes `magnitude` plus `magnitude_unit: sigma | absolute` (default `sigma`).
- Ramps (drifts, noise growth, decoupling) include the sample at `end`, so a drift reaches its full magnitude there. With `hold_after: true` the effect persists and the label `end` is null.
- `first_detectable` is the first time the injected effect exceeds 1 sigma, computed from the model and not from any detector. It is null if that never happens.
- Label fields for odd types: `dropout` has no magnitude. `noise_growth` magnitude is the final factor (no sigma value). `stuck_sensor` magnitude is the frozen value minus baseline. `decoupling` has no magnitude.
- Severity: isolated spike and dropout under 5 s are "low". Everything else is "high".
- In `data.csv` an empty cell means no sample at that time. A dropout is written as `nan`.

Batch (Phase 2):
- Scenario categories use exact counts: clean (25% of all runs), multi (25% of all runs), hard (20% of the non-clean runs), single (the rest). Primary anomaly types are assigned round-robin.
- Hard cases: very slow drift, spikes of 3.5 to 5 sigma, a step shortly after a spike, stuck sensor.
- Every decoupling comes with a driver anomaly (a held drift or step on the source channel), otherwise it would be invisible.
- Each scenario has a unique seed derived from the master seed. Splits are assigned per scenario from the master seed. Anomaly types missing from a split are reported in the summary, not hidden.

## Known limits

- With `--n 200` a few anomaly types can be absent from the 20-run val or test split (master seed 1: `transient_decay` in val, `dropout` in test). A larger `--n` or a stratified split would fix this.
