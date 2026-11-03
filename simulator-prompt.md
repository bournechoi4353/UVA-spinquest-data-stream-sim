# Claude Code Prompt: Synthetic Data Stream Simulator for an Anomaly-Monitoring LLM

Paste this whole file into Claude Code at the root of a new, empty project folder.

---

## ROLE AND GOAL

You are helping me build the data foundation for a project: a small fine-tuned LLM that answers questions about abnormalities in a multi-channel sensor data stream (inspired by detector and target monitoring for a nuclear physics experiment). The LLM never sees raw data. Plain code detectors watch the stream, write conclusions into a state store, and the LLM only phrases answers from a short text summary.

Your job in this prompt is ONLY the first piece: a **seeded, configurable, synthetic multi-channel data stream simulator** that injects known anomalies and records exact ground truth. Later phases (detectors, summary builder, LLM training data, Discord bot) are out of scope here, except the optional reference detector check in Phase 6.

Hard rules:
- Use NO real experiment data and make NO network calls. Everything is synthetic.
- Channel names are generic placeholders (see config). Do not claim they match any real instrument.
- Do not use em dashes anywhere: not in code comments, docstrings, README text, or printed output. Use commas, colons, or parentheses.
- Pin every dependency version in `requirements.txt`. Python 3.11+. Allowed libraries: numpy, pandas, pyyaml, matplotlib, pytest. Nothing else without asking me.
- All randomness must come from `numpy.random.Generator` seeded from the scenario config. Same seed and config must give byte-identical output.
- Work in gated phases. At the end of each phase, STOP, show me what exists, run the tests, summarize in plain language, and wait for my approval before starting the next phase.
- Write clear docstrings. I am a student learning this, so keep the code readable over clever.

---

## PROJECT LAYOUT

```
anomaly-sim/
  README.md
  requirements.txt
  configs/
    healthy.yaml
    mixed_demo.yaml
  sim/
    __init__.py
    channels.py        # baseline signal models per channel
    anomalies.py       # anomaly injectors
    scenario.py        # config loading and validation
    stream.py          # generator and replay interface
    labels.py          # ground truth event records
    io.py              # CSV and JSON writers and readers
    plotting.py        # inspection plots
    batch.py           # many randomized scenarios (Phase 2)
    reference_detectors.py   # Phase 6 only
  scripts/
    make_run.py
    plot_run.py
    make_batch.py
    score_detectors.py   # Phase 6 only
  tests/
  out/                   # generated data, gitignored
```

---

## PHASE 1: CORE SIMULATOR

### Healthy signal model (channels.py)

Each channel has:
- `name`
- `baseline`: healthy mean
- `sigma`: healthy point-to-point noise standard deviation
- `sample_rate_hz`: default 1.0, allow different rates per channel
- optional `ar1_phi`: autocorrelation of the noise (default 0.0, meaning white noise). When phi is nonzero, scale the innovation so the stationary standard deviation still equals `sigma`.
- optional `slow_wander`: amplitude and period of a harmless slow sinusoidal wobble (default 0), so that healthy data is not perfectly flat
- optional `couples_to`: another channel name plus a `gain`, meaning this channel's healthy value follows `baseline + gain * (other_channel_deviation_from_its_baseline) + own noise`. This is used for the decoupling anomaly later.

Default channels in the example configs: `polarization` (baseline 41.0, sigma 0.15), `beam_current` (5.0, 0.05), `rf_power` (100.0, 0.4), `target_temp` (1.0, 0.01), and `nmr_signal` (sampled at 0.2 Hz, baseline 2.5, sigma 0.03, couples to `polarization` with gain 0.05). Make all of this configurable. Different sample rates mean the stream must merge channels by timestamp (see stream.py).

### Anomaly injectors (anomalies.py)

Each anomaly is applied on top of the healthy signal for one channel over a time window, and each one emits a ground truth label. Implement all of these:

1. `spike`: a single-sample jump of `magnitude_sigma` standard deviations (positive or negative).
2. `spike_burst`: several spikes within a short window (count, spacing), to model instability.
3. `step_shift`: at `start`, the level jumps by `magnitude` (in sigma units or absolute units) and stays there until `end` (or forever).
4. `linear_drift`: the level changes linearly, total change `magnitude` over the window, optionally holding the final offset afterward.
5. `exponential_drift`: slow start, accelerating growth, with a configurable time constant.
6. `noise_growth`: the noise sigma grows from 1x to `factor`x over the window while the mean stays put.
7. `transient_decay`: a jump at `start` that decays exponentially back to baseline with a given time constant.
8. `dropout`: samples become NaN for the window (missing data).
9. `stuck_sensor`: the channel freezes at its last value for the window (flat line with zero noise).
10. `decoupling`: for a coupled channel, the coupling gain slowly decays toward zero over the window, so each channel looks fine alone but the relationship breaks.

Anomalies must compose: several can be active on one channel at once and on different channels at once.

### Ground truth labels (labels.py)

Every injected anomaly produces a record:

```
{
  "id": "ev_0007",
  "channel": "polarization",
  "type": "linear_drift",
  "start": "2026-01-01T09:41:00",
  "end": "2026-01-01T12:20:00",     # null if it never ends
  "magnitude": -3.5,                # in channel units
  "magnitude_sigma": -23.3,         # in healthy sigma units
  "params": {...},                  # all parameters used
  "severity": "high" | "low"        # drift, shift, noise_growth, decoupling, stuck_sensor = high; isolated spike, dropout under 5 s = low; burst = high
}
```

Also record `first_detectable`: the earliest time the anomaly's effect exceeds 1 sigma (computed from the injected model, not from any detector). For drifts this is later than `start`, and it matters for scoring detection delay later.

### Scenario config (scenario.py)

YAML with: `seed`, `start_time` (ISO string), `duration_seconds`, `channels` (list), `anomalies` (list, each with channel, type, start offset in seconds, and type-specific parameters). Validate everything with clear error messages (unknown channel, anomaly outside duration, negative durations, and so on). Provide two example configs:
- `healthy.yaml`: 6 hours, no anomalies, for calibration.
- `mixed_demo.yaml`: 6 hours with a slow linear drift on polarization starting at hour 2, a few isolated spikes on beam_current, a step shift on rf_power at hour 4, and a late decoupling of `nmr_signal`.

### Stream and replay interface (stream.py)

- `generate(config) -> StreamResult` that produces full arrays plus labels (fast path for batch generation).
- `replay(config_or_files, speed=None)` returning an iterator of `(timestamp, {channel: value})` events in time order, merging channels of different sample rates (a channel only appears in an event when it has a sample at that time). `speed=None` means as fast as possible. A float means real time multiplied by that factor, using `time.sleep` between events. This iterator is what the detector loop will later consume, so keep the interface simple and stable.
- Never let the replay iterator expose the labels. Ground truth is available only through the separate labels API, so a detector cannot cheat.

### Output (io.py, scripts/make_run.py)

`python scripts/make_run.py configs/mixed_demo.yaml` writes to `out/<run_name>/`:
- `data.csv` (columns: timestamp, then one column per channel, empty cell where a channel has no sample at that time)
- `labels.json`
- `config_used.yaml` (copy of the exact config, with the seed)

### Phase 1 tests (tests/)
- Same seed gives identical `data.csv` and `labels.json`.
- Healthy channel sample mean and standard deviation are within statistical tolerance of the config (use a long run and a sensible tolerance).
- AR(1) noise keeps the target stationary standard deviation.
- Each anomaly type produces the expected magnitude at the expected time (for example, linear drift reaches `magnitude` at `end`, step shift offset matches, dropout yields NaNs only inside the window).
- Labels match injected anomalies exactly.
- Config validation rejects bad input with readable messages.

**STOP after Phase 1. Show me a sample of the CSV, the labels for the mixed demo, and the test results. Wait for approval.**

---

## PHASE 2: BATCH GENERATION FOR TRAINING DATA

Build `batch.py` and `scripts/make_batch.py` to generate many randomized scenarios:

- Randomize number of channels used (subset of the defaults), baselines and sigmas within ranges, scenario length (1 to 12 hours), and a random set of anomalies drawn from all types with random parameters.
- Include a configurable fraction of **clean runs with no anomalies** (default 25 percent) and a fraction with **multiple simultaneous anomalies** (default 25 percent), and some **hard cases**: very slow drifts that barely cross threshold, spikes that look like noise, step shifts right after a spike, and stuck sensors.
- Each scenario gets its own seed derived from a master seed and its index. Save each run with its config and labels, plus one `manifest.json` listing every run, its seed, and which anomaly types it contains.
- Split into `train`, `val`, `test` at the **scenario seed level** (default 80, 10, 10), so no scenario leaks across splits. Record the split in the manifest. Never generate a test scenario from a seed used in train.
- Command line options: `--n`, `--master-seed`, `--out`, `--clean-frac`, `--hard-frac`.
- Report a summary table at the end: counts per anomaly type per split, so I can check balance.

Tests: splits are disjoint, the manifest counts match the files on disk, the clean fraction is respected within tolerance, and the same master seed reproduces the same batch.

**STOP after Phase 2 and wait for approval.**

---

## PHASE 3: INSPECTION PLOTS

Build `plotting.py` and `scripts/plot_run.py`:
- One subplot per channel over time, with healthy baseline as a dashed line and a shaded +/- 3 sigma band.
- Shade each labeled anomaly window and annotate its type. Use different colors for different types, and mark `first_detectable` with a small tick.
- Missing data shown as gaps.
- Optional overlay of a second channel on one axis for the decoupling anomaly (scatter plot of the coupled pair, healthy vs anomalous period colored differently).
- Save PNGs to `out/<run>/plots/`. No interactive windows.

**STOP after Phase 3. Show me the plot for `mixed_demo` and wait for approval.**

---

## PHASE 4: README

Write a README explaining: what the project is, how to run each script, every config field, every anomaly type with its parameters and a one-line physical intuition, the label schema, the replay interface contract, and how determinism works. Include a short "How this plugs into the rest of the pipeline" section describing (without implementing) the detectors, state store, summary builder, and LLM. No em dashes.

**STOP after Phase 4 and wait for approval.**

---

## PHASE 5 (OPTIONAL, ONLY IF I SAY GO): REFERENCE DETECTORS AND SCORING

Implement a minimal reference detector set in `reference_detectors.py` so I can verify the simulator produces data that real detectors can and cannot catch. Per channel, calibrate `target` and `sigma` from a healthy run (use the robust median and MAD estimate: sigma = 1.4826 * median(|x - median|)).

1. **Spike detector:** z = (x - mean) / sigma. Flag when |z| exceeds the threshold (default 5). The adaptive mean updates slowly (alpha 0.01) and does NOT update on flagged points.
2. **CUSUM** against the FROZEN healthy target, with slack = 0.5 sigma and threshold = 5 sigma by default. Input deviations are clipped to +/- 3 sigma so one spike cannot trigger a drift alarm. Track `since`, the time the score first rose above zero.
3. Handle missing samples (NaN) by skipping them without resetting state.

Then `scripts/score_detectors.py`:
- Run the detectors on a batch's test split through the `replay` interface only (never reading labels during detection).
- Match detector events to ground truth labels and report, per anomaly type: detection rate, **detection delay** (alarm time minus `first_detectable`), and false alarms per hour on clean runs.
- Print a table. Flag any anomaly type with detection rate under 90 percent so I know which simulated anomalies the simple detectors miss (expected: stuck sensor and decoupling, which need extra logic).

**STOP and report.** Do not tune thresholds silently. Show me the numbers and wait.

---

## DEFINITION OF DONE

- `pytest` passes.
- `python scripts/make_run.py configs/mixed_demo.yaml` works and is reproducible.
- `python scripts/make_batch.py --n 200 --master-seed 1` produces a balanced, split batch with a manifest.
- Plots clearly show the injected anomalies.
- No em dashes anywhere in the repo (run a grep check and report the result).

Start with Phase 1. First, restate the plan in a few lines and list any questions or assumptions, then begin.
