# Claude Code Prompt: Align the Simulator With the Real ptdb Layout and Add a Read-Only Data Profiler

Paste this whole file into Claude Code at the root of the existing simulator repo (the one built from the earlier simulator prompt).

BEFORE YOU PASTE: Section 3 holds what I know about the real data. Replace every `[FILL IN: ...]` with what you know. Anything left as `[FILL IN]` becomes a TODO that Claude Code must not guess.

---

## 1. ROLE AND GOAL

The repo already contains a seeded multi-channel simulator with ground-truth labels, batch generation, plots, and tests. The project behind it is a small fine-tuned LLM that answers questions about abnormalities in a sensor stream. Plain-code detectors write conclusions to a state store, and the model only phrases answers from a short text summary.

The real data lives in a MySQL database named `ptdb` (described in section 3). The simulator so far uses generic placeholder channels. Your job in this prompt:

1. Describe the real layout in a registry file (R1).
2. Teach the simulator to produce data shaped like the real tables, and to write it into a local fixture database with the same table and column names (R2).
3. Write a read-only profiler that I will run myself against the real database and that outputs aggregate statistics only (R3).
4. Fit the simulator to the profiler's output and check the fit (R4).

Out of scope here: the live MySQL ingestion reader, the detectors, the state store, the summary builder, the model, and the Discord bot. Do not start them.

---

## 2. HARD RULES

General (carried over from before):
- No em dashes anywhere: not in code, comments, docstrings, README text, config files, or printed output. Use commas, colons, or parentheses.
- Python 3.11+. Pin every dependency in `requirements.txt`. New allowed packages: `pymysql` (R3 only, pinned). SQLite uses the standard library. Anything else needs my approval first.
- All randomness comes from seeded `numpy.random.Generator` objects. Same seed and config give identical output.
- Work in gated phases R1 to R4. At the end of each phase: run the full test suite, summarize in plain language, show me the key outputs, and STOP until I approve. After I approve, commit once with a message like `Phase R1: real layout registry`.
- Keep every existing test passing. Do not change the existing label schema or replay contract except where a phase says so.
- Write readable code with docstrings. I am a student learning this.

Data safety (hard rules, no exceptions):
1. **Do not connect to the live database from this session.** Write code and tests against local fixtures only. I run anything that touches the real database myself, on Shannon.
2. **The profiler outputs aggregates only.** Never write raw rows, per-row values, or per-row timestamps to disk, logs, or the console. The only individual timestamps allowed are the start and end of the 20 largest gaps per table.
3. **Credentials come from environment variables only:** `PTDB_HOST`, `PTDB_PORT`, `PTDB_USER`, `PTDB_PASSWORD`, `PTDB_DATABASE`. Never put them in code, config, tests, logs, or git. Create `.env.example` with the variable names and no values, and add `.env` to `.gitignore`.
4. **Real profiler output is never committed.** Add `out/` and `out/profile/` to `.gitignore`, and add a test that fails if those entries are missing.
5. **Read-only by construction.** The database adapter must allow only statements that start with `SELECT`, `SHOW`, or `SET` (for the session time zone and read-only flags), reject everything else, and have a test for the rejection. On connect, run `SET SESSION TRANSACTION READ ONLY` and `SET time_zone = '+00:00'`.
6. **Be gentle on the database:** chunked reads by `id` range, a per-query timeout set through a session variable chosen by server version (`SET SESSION max_statement_time = 30`, in seconds, on MariaDB; `SET SESSION max_execution_time = 30000`, in milliseconds, on MySQL), where the adapter detects which one from `SELECT VERSION()` and has a test for both branches, a configurable pause between queries, and no full-table `COUNT(*)` (use `MIN(id)` and `MAX(id)`). Provide `--dry-run` that prints the planned queries and runs nothing.
7. **Synthetic data wearing real names must be labeled.** Any simulated output that uses real channel names carries `synthetic: true` in its metadata, plot titles say "SIMULATED", and the README says so.
8. At the end of R3, remind me that the profiler output must be reviewed by me before it is shared with anyone, including in a chat.

---

## 3. WHAT I KNOW ABOUT THE DATA (I edit this section before pasting)

### 3.1 Database and time

- MySQL database `ptdb`. Read access only. Shannon can reach it.
- `Timestamp` columns are MySQL `timestamp` type (stored in UTC, shown in the session time zone) in every table except `sensor_data` and `stress_test_data`, which use `datetime` (no time zone). The server's `time_zone` setting is `SYSTEM`. The actual system zone is [FILL IN: result of `SELECT @@system_time_zone`].
- Every table has an auto-increment `id` (int) as primary key and a secondary index on `Timestamp`. Engine InnoDB, charset latin1. Confirmed with `SHOW CREATE TABLE` for Labjack, Flow_Rates, and Lakeshore_Fridge_Temp. The other tables: [FILL IN: assumed the same, or paste their `SHOW CREATE TABLE`].
- `Timestamp` is `timestamp NOT NULL DEFAULT current_timestamp()` with no fractional seconds and no `ON UPDATE` clause. That most likely makes it the row insert time at 1 second resolution, not the instrument's measurement time. Rows 1.8 s to 3.6 s apart therefore often share a second, so always order by `id`, never by `Timestamp` alone.
- Every value column is a single-precision `float DEFAULT NULL`, so a failed read may be stored as NULL or as a sentinel number. Sentinel convention: [FILL IN, or "unknown"].
- In all three tables the next AUTO_INCREMENT value equals the row count plus 1 (17,770,894 for Labjack, 13,902,881 for Flow_Rates, 8,885,452 for Lakeshore_Fridge_Temp). That suggests ids run from 1 to N with no deleted rows. The profiler will confirm that `MIN(id)` is 1.
- The `current_timestamp()` style in the table definitions suggests the server may be MariaDB rather than MySQL. Version: [FILL IN: result of `SELECT VERSION()`]. All code must work on both.
- Rows seen so far (times as displayed in the server session zone):

| Table | Rows | First timestamp | Last timestamp |
| --- | --- | --- | --- |
| NMR | 0 | none | none |
| Flow_Rates | 13,902,880 | 2025-08-15 11:11:20 | 2026-08-17 17:29:08 |
| Labjack | 17,770,893 | 2025-08-15 11:11:20 | 2026-08-17 17:29:08 |
| Lakeshore_Fridge_Temp | 8,885,451 | 2025-08-15 11:11:20 | 2026-08-17 17:29:08 |
| IVC, Lakeshore_Magnet_Temp, Lakeshore_Target_Stick, MaxiGauge, QT, sensor_data, stress_test_data | [FILL IN] | [FILL IN] | [FILL IN] |

- Derived from the counts, assuming the logger never paused: average spacing is about 2.28 s for Flow_Rates, 1.79 s for Labjack, 3.57 s for Lakeshore_Fridge_Temp. Labjack has almost exactly twice the rows of Lakeshore_Fridge_Temp (17,770,893 against 8,885,451). Real gaps are unknown until the profiler runs.
- The newest rows are from 2026-08-17. Is the logger still writing? [FILL IN]
- A `COUNT(*)` over four of these tables took 23 seconds, so full scans are expensive.

### 3.2 Tables and columns (from `information_schema`, spelling kept exactly)

Every table has `id` and a time column in addition to the columns below. Value columns are `float` unless noted.

- **Flow_Rates** (time column `Timestamp`): `seperator_flow`, `magnet_flow`, `main_flow`, `microwave_flow`, `heat_exchanger_flow`
- **IVC** (`Timestamp`): `ivc_pressure`
- **Labjack** (`Timestamp`): `root_exhaust_pressure`, `buffer_pressure`, `magnet_pressure`, `purifier_inlet_pressure`, `fridge_vapor_pressure`, `thermocouple`, `magnet_bottom_temperature`, `magnet_top_temperature`
- **Lakeshore_Fridge_Temp** (`Timestamp`): `fridge_target_top_up_temperature`, `fridge_target_top_up_center_temperature`, `fridge_target_top_down_temperature`, `fridge_target_bottom_up_temperature`, `fridge_target_bottom_up_center_temperature`, `fridge_target_bottom_down_temperature`, `fridge_target_top_cernox_temperature`, `fridge_target_bottom_cernox_temperature`
- **Lakeshore_Magnet_Temp** (`Timestamp`): `magnet_channel_1` to `magnet_channel_8` (unnamed channels)
- **Lakeshore_Target_Stick** (`Timestamp`): `target_stick_buffle_top_temperature`, `target_stick_buffle_bottom_temperature`, `target_stick_seperator_top_temperature`, `target_stick_seperator_bottom_temperature`, `target_stick_heat_exchanger_top_temperature`, `target_stick_heat_exchanger_bottom_temperature`, `target_stick_annealing_plate_bar_temperature`, `target_stick_annealing_plate_top_temperature`
- **MaxiGauge** (`Timestamp`): `maxigauge_seperator_inlet_pressure`, `maxigauge_upper_roots_pressure`, `maxigauge_channel_3` to `maxigauge_channel_6` (channels 3 to 6 unnamed)
- **NMR** (`Timestamp`): `run_number` (int), `measurement_type` (text), `peak_amp`, `peak_center`, `beam_on` (int), `rf_level`, `if_atten`, `he_temperature`, `he_pressure`, `nmr_channel` (int), `temperature`, `calibration_constant`, `polarization`, `polarization_std`, `snr`, `step_width`, `center_freq`, `freq_span`, `area`, `phase_voltage`, `tune_voltage`
- **QT** (`Timestamp`): `fc501_ai`, `fc501_out`, `fc502_ai`, `fc502_out`, `lit501_ai`, `pt501_ai`, `pt502_ai`, `pt503_ai`, `pt504_ai`, `ait501_ai`, `ti501_ai`, `ti502_ai`, `ti503_ai`, `ti504_ai`, `ti505_ai`, `ti523_ai`
- **sensor_data** (time column `timestamp`, type `datetime`): `value`
- **stress_test_data** (time column `Timestamp`, type `datetime`): `col1` to `col5`

The whole schema has 109 columns (this is a test target in R1).

### 3.3 Scope of the first version

- **Included feeds (21 channels):** Flow_Rates (5), Labjack (8), Lakeshore_Fridge_Temp (8).
- **Deferred, listed in the registry with `include: false` and a reason:** IVC, Lakeshore_Magnet_Temp, Lakeshore_Target_Stick, MaxiGauge, QT, NMR (empty, polarization source unknown).
- **Excluded for now:** `sensor_data` (one value column and no channel name, purpose unknown) and `stress_test_data` (presumably test data).

### 3.4 Unknowns (leave as TODOs, never guess)

- Where the polarization readings live: [FILL IN]
- Units for every channel: [FILL IN, or "unknown"]
- Names for the unnamed channels (`magnet_channel_1` to `8`, `maxigauge_channel_3` to `6`): [FILL IN]
- What `QT`, `IVC`, and `sensor_data` hold: [FILL IN]
- Which channel pairs are physically coupled (for the decoupling checks): [FILL IN, or "take from the profiler's correlation list in R3"]
- Whether `buffle` in two column names means "baffle": [FILL IN]
- Known incidents, outages, or maintenance windows between 2025-08 and 2026-08: [FILL IN, or "none known"]

---

## 4. REQUIREMENTS THIS WORK MUST SERVE (from the project requirements doc)

- ING-3: the reader yields `(timestamp, {channel: value})` events in the same form as the simulator's replay interface.
- ING-4: a config file maps table, time column, and channel columns to channel names.
- ING-5 and ING-16: all times are converted to UTC. `timestamp` columns are stored in UTC, `datetime` columns have no zone, and the zone assumption is recorded per table.
- ING-6: late, out-of-order, and duplicate rows must be handled, and NULLs are missing samples that do not reset detector state.
- ING-7: each channel or feed can have its own sample rate.
- ING-11: a MySQL-shaped test fixture (a small local database or recorded rows) lets tests run without the live database.
- ING-12: the auto-increment `id` is the polling watermark where it is monotonic.
- ING-13: a channel registry gives display names, units, and groups.
- ING-17 and ING-18: the connection time zone is UTC, history is read in chunks by `id`, and calibration and backfill are vectorized.
- ING-19: order and page by `id`, never by `Timestamp` alone, because `Timestamp` is the insert time at 1 second resolution.
- ING-20: choose the query timeout by server version (MariaDB or MySQL).

Channel names inside the system are `<table>.<db_column>` (for example `Labjack.fridge_vapor_pressure`).

---

## 5. PHASES

### 5.0 DATA FLOW OVERVIEW (applies to every phase)

Every phase plugs into one pipeline. Data only ever moves left to right, and the real database is touched only by the profiler, run by me on Shannon.

```
configs/ptdb_layout.yaml ──> sim/registry.py ──> Registry (tables, feeds, channels)
                                                     │
                     ┌───────────────────────────────┼─────────────────────────────┐
                     v                               v                             v
        scripts/make_layout_scenario.py     sim/db_writer.py              sim/profiler/
        (wide feeds + labels, R2)           (SQLite fixture, R2)          (aggregates only, R3)
                     │                               │                             │
                     v                               v                             v
        replay_layout() events            fixture.sqlite (same names       out/profile/ptdb_profile.json
        (timestamp, {channel: value})      as ptdb)                                 │
                                                     ^                             v
                                                     └──── sim/from_profile.py <───┘ (R4, fit and compare)
```

Shared conventions for reading data in every phase:
- **Channel key:** always `<table>.<db_column>` (for example `Labjack.fridge_vapor_pressure`). Display names are for humans only and never used as keys.
- **Order:** rows are always read `ORDER BY id`. `Timestamp` is a label on the row, not a sort key (ING-19).
- **Time:** every timestamp inside Python is a timezone-aware UTC `datetime` (or a UTC epoch float in numpy arrays). Conversion happens once, at the edge where data is read. `timestamp` columns are UTC because the session zone is `+00:00`. `datetime` columns carry no zone, so the reader attaches the zone recorded for that table in the layout (`time_zone_assumption`, default `UTC`, flagged as an assumption).
- **Missing data:** a NULL value is a missing sample for that channel only. An absent row (a gap in time) is a missing sample for every channel in the feed. These are different and are never merged.
- **Chunked reads:** any code that reads a table pages by `id`: `WHERE id > :last_id ORDER BY id LIMIT :n`, then sets `last_id` to the largest id seen. This is the same watermark pattern the future live reader will use (ING-12), so the fixture and profiler exercise it now.

### PHASE R1: LAYOUT AND REGISTRY (no database)

1. Create `configs/ptdb_layout.yaml` from section 3. For every table: `name`, `id_column`, `time_column`, `time_type` (`timestamp` or `datetime`), `timestamp_resolution_s` (1 for the three checked tables, null for the rest until known), `include` (true only for the three included feeds, otherwise false with a `reason`), `nominal_period_s` (2.28, 1.79, and 3.57 for the included feeds, null for the rest, with a comment that these are averages assuming no gaps), and a `channels` list.
2. Each channel has:
   - `db_column`: the exact column name, typos kept (`seperator`, `buffle`).
   - `display_name`: a readable name. Fix only the clear misspelling `seperator` to `separator`. Leave `buffle` as is and add a TODO note.
   - `unit`: null for everything unless section 3.4 gives one.
   - `group`: `temperature`, `pressure`, `flow`, or `other`, taken from the column name, with `group_inferred: true`.
   - `role`: `sensor`, or `controller_output` for the QT columns ending in `_out`.
   - `registry_status`: `named`, `inferred`, or `unnamed` (use `unnamed` for `magnet_channel_N`, `maxigauge_channel_3` to `6`, and `col1` to `col5`).
3. Create `sim/registry.py` to load and validate the layout. Reject unknown keys, duplicate `db_column` values within a table, an included feed with no time column or period, and any layout that does not list every column from section 3.2.
4. Create `scripts/show_registry.py` that prints the included channels in a table and a count.
5. Tests: the layout has 109 columns in total (including `id` and time columns), exactly 21 included channels, every included channel has a nonempty display name, the unnamed channels are flagged, and the validator rejects each bad case above.
6. Create `.env.example`, and make sure `.gitignore` contains `.env`, `out/`, and `out/profile/`, with a test for it.

**How the data is read in R1:** no sensor data is read. The only input is `configs/ptdb_layout.yaml`, read with `yaml.safe_load` and validated by `sim/registry.py`. Each table also gets a `time_zone_assumption` field (`UTC` for `timestamp` tables, `UTC` with `assumed: true` for the two `datetime` tables until section 3.1 is filled in).

**How it works:**
- `load_registry(path) -> Registry` parses the YAML into frozen dataclasses (`Table`, `Channel`) and runs every validation rule before returning. A bad layout raises a `RegistryError` that names the table and column at fault, so a typo never survives silently.
- The `Registry` is the single source of truth that every later phase reads instead of hardcoding names. It exposes:
  - `registry.included_tables()` : the three included feeds.
  - `registry.channels(table)` : the channel list for one table, in column order.
  - `registry.channel("Labjack.fridge_vapor_pressure")` : look up one channel by its key.
  - `registry.select_columns(table)` : the exact column list (`id`, time column, value columns) that any SQL query for that table must use, so queries are built from the registry and never typed by hand.
- Example use:
  ```python
  from sim.registry import load_registry
  reg = load_registry("configs/ptdb_layout.yaml")
  for t in reg.included_tables():
      print(t.name, t.nominal_period_s, [c.key for c in reg.channels(t.name)])
  ```
- `scripts/show_registry.py` is just a printed view of the same object: table, key, display name, group, status, unit (or `unknown`).

**STOP after R1.** Show me the registry table and the test results.

### PHASE R2: WIDE FEEDS AND A FIXTURE DATABASE

1. **Wide feeds.** A feed is a table: its channels share one timestamp series. Add a scenario generator `scripts/make_layout_scenario.py --hours 6 --seed 1 --anomalies mixed` that builds a scenario over the included feeds using the registry. Until a profile exists (R4), use plainly synthetic defaults (baseline 100, sigma 1 for every channel), overridable per channel, and mark the output `synthetic: true`.
2. **Realistic irregularity (healthy, not anomalies):** timestamps at whole-second resolution (`timestamp_resolution_s`, default 1) so rows closer than a second apart can share a timestamp, a small insert-time jitter per feed (`jitter_s`, default 5 percent of the period, bounded), and occasional single-row gaps at a configurable rate. Rows are ordered by `id`, and `Timestamp` is the insert time.
3. **New anomaly `feed_outage`:** every channel in a feed stops for a window, and the rows are absent (not NaN). Label severity is "high", and `first_detectable` is the first time the gap exceeds 3 times the nominal period. All existing anomaly types keep working on wide feeds.
4. **Cross-feed coupling:** the existing coupling feature may link channels in different feeds. The layout may declare couplings, but the default is none. Do not invent physical couplings.
5. **Replay:** one feed row yields one event with all of that feed's channels at that time. Keep labels hidden from the replay interface.
6. **Fixture database writer** `sim/db_writer.py` and `scripts/make_fixture_db.py`: write a simulated run into a local SQLite file with the same table and column names as the layout, including `id` and the time column. `id` is assigned in insertion order. Add options:
   - `--late-fraction` inserts a small share of rows out of time order (they get later ids). Real rows probably rarely arrive out of order, because `Timestamp` is assigned at insert, so treat this as a stress test.
   - `--dup-fraction` inserts a small share of duplicate rows.
   - `--null-fraction` turns a small share of values into NULL.
   These exist so the future ingestion reader can be tested against ING-6.
7. Tests: writing a run to SQLite and reading it back reproduces the data exactly, `feed_outage` produces no rows in its window, jitter stays within bounds, the late, duplicate, and NULL options are deterministic for a seed, and labels use realized times.

**How the data is read in R2:** three ways, all on synthetic data.

1. **Files on disk.** `make_layout_scenario.py` writes `out/layout/<run_name>/`:
   - `feeds/<Table>.csv` : one file per feed, columns `id`, the time column (UTC, `YYYY-MM-DD HH:MM:SS`), then the value columns with their real `db_column` names. An empty cell is NULL.
   - `labels.json` : ground truth in the existing label schema, with channels as `<table>.<db_column>` and feed-level anomalies (`feed_outage`) using the table name as the target.
   - `meta.json` : seed, config used, `synthetic: true`, and the registry version.
   Read a feed with `pandas.read_csv(path, parse_dates=[time_col])` and localize to UTC.
2. **Replay.** `sim.replay_layout(config_or_run_dir)` yields `(timestamp, {channel_key: value})` in the same shape as the existing `replay()` (ING-3). One event per feed row. Events from different feeds are merged by timestamp, with ties broken by table name then `id`, so the order is deterministic. A NULL appears as a missing key, not as `nan`. Labels are never reachable from the replay object.
   ```python
   from sim import replay_layout
   for ts, values in replay_layout("out/layout/demo"):
       ...  # values like {"Labjack.buffer_pressure": 101.2, "Labjack.thermocouple": 99.7, ...}
   ```
3. **SQLite fixture.** `make_fixture_db.py` writes `out/layout/<run_name>/fixture.sqlite` with one table per included feed, using the real table and column names. The `id` column is `INTEGER PRIMARY KEY` (insertion order), the time column is `TEXT` in UTC `YYYY-MM-DD HH:MM:SS` (SQLite has no timestamp type, and this matches what MySQL returns with the session zone at `+00:00`), and value columns are `REAL` with NULLs allowed. There is an index on the time column, matching ptdb. Read it exactly as the real database will be read:
   ```sql
   SELECT id, Timestamp, buffer_pressure, ... FROM Labjack WHERE id > ? ORDER BY id LIMIT 5000;
   ```

**How it works:**
- For each included feed, the generator first builds the row timestamps: a regular grid at `nominal_period_s`, plus bounded jitter, minus single-row gaps and `feed_outage` windows, then floored to `timestamp_resolution_s`. Ids are assigned after that, in order.
- Each channel's values are generated on that feed's row times using the existing channel model (baseline, sigma, AR(1), wander, couplings), and existing anomalies are injected per channel as before. Labels are computed from the realized (jittered, floored) row times.
- The fixture writer then applies the stress options in a fixed order with their own seeded generator: NULLs first, then duplicates (a copy of an existing row inserted later with a new id), then late rows (a row moved later in insertion order so its id is larger than rows with later timestamps). Because the order and seed are fixed, the same options always give the same file.
- A round-trip reader `sim/db_writer.read_fixture(path, table)` pages by `id` and returns a DataFrame, which is what the round-trip tests compare against the in-memory run.

**STOP after R2.** Show me a plot of a wide scenario with a feed outage, plus the test results.

### PHASE R3: READ-ONLY PROFILER

Build `scripts/profile_db.py` and `sim/profiler/` with two adapters (a `MySQLAdapter` for the real database and a `SQLiteAdapter` for tests) behind one interface, with the read-only guard from rule 5 in front of both.

**Sampling plan (no full scans):**
- For each included table, read `MIN(id)` and `MAX(id)`, then sample `--chunks` (default 200) evenly spaced id ranges of `--chunk-rows` (default 5,000) consecutive rows. Consecutive rows are needed for spacing and autocorrelation.
- Cross-channel correlation uses three 24-hour windows (start, middle, end of the available range) aligned to a 60 s grid with last-observation-carried-forward.
- Expose `--tables`, `--since`, `--until`, `--pause-seconds`, and `--dry-run`.

**Per table (aggregates only):**
- `id_min`, `id_max`, row estimate from the ids, whether `id_min` is 1, the share of sampled consecutive id pairs that differ by exactly 1, and the first and last timestamp.
- Server check, once: `VERSION()`, `@@system_time_zone`, `NOW()`, and `UTC_TIMESTAMP()`.
- Share of sampled consecutive id pairs where the timestamp goes backward (ordering check), and share of duplicate timestamps.
- Spacing between consecutive rows: percentiles p1, p5, p50, p95, p99, and max; the number of gaps larger than 5 times the median; and the start and end time of the 20 largest gaps. Spacing is in whole seconds, so also report the share of consecutive rows with spacing 0, 1, 2, and more than 2 seconds.

**Per channel (aggregates only):**
- Sample size, null fraction, min, max, mean, standard deviation, median, and MAD-based sigma (1.4826 times the median absolute deviation).
- Short-term noise: the MAD sigma of the residual after subtracting a centered rolling median (window 61 samples), its lag-1 autocorrelation, and the slow-component share (1 minus residual variance divided by total variance).
- Stuck-value indicators: share of consecutive identical values and the longest run of identical values.

**Cross-channel:** the top 30 channel pairs by absolute correlation of first differences, and the top 30 by absolute correlation of levels, using `<table>.<db_column>` names.

**Output:** `out/profile/ptdb_profile.json` and a human-readable `ptdb_profile.md`. The JSON schema is an allowlist: no key outside it, and no list longer than 20 items except the percentile lists and the 30-pair correlation lists. Put a banner at the top of both files: "AGGREGATES ONLY. Review before sharing."

**How the data is read in R3:** this is the only code that will ever touch ptdb, and only when I run it.

1. **Connect.** `MySQLAdapter` reads `PTDB_HOST`, `PTDB_PORT`, `PTDB_USER`, `PTDB_PASSWORD`, `PTDB_DATABASE` from the environment (it fails with a clear message if any are missing, and never prints the password). Right after connecting it runs, in order:
   ```sql
   SET SESSION TRANSACTION READ ONLY;
   SET time_zone = '+00:00';
   SELECT VERSION();                         -- picks the timeout statement below
   SET SESSION max_statement_time = 30;      -- MariaDB, seconds
   SET SESSION max_execution_time = 30000;   -- MySQL, milliseconds (only one of the two runs)
   SELECT VERSION(), @@system_time_zone, NOW(), UTC_TIMESTAMP();
   ```
   `SQLiteAdapter` opens the fixture file in read-only mode (`file:...?mode=ro`) and skips the server statements.
2. **Bounds, using the primary key only.** `SELECT MIN(id), MAX(id) FROM <table>` (an index lookup, not a scan), then `SELECT Timestamp FROM <table> WHERE id = :min` and the same for `:max` to get the first and last time. With `--since` or `--until`, the id bounds come from `SELECT MIN(id) FROM <table> WHERE Timestamp >= :since` (and `MAX(id) ... <= :until`), which uses the `Timestamp` index.
3. **Sample chunks.** For each of the `--chunks` evenly spaced starting ids `a`:
   ```sql
   SELECT id, Timestamp, <value columns from the registry> FROM <table>
   WHERE id >= :a AND id < :a + :chunk_rows ORDER BY id;
   ```
   Then sleep `--pause-seconds`. Each chunk is read into a numpy array, its statistics are folded into running accumulators, and the raw chunk is deleted before the next query. Raw rows exist only in memory, one chunk at a time.
4. **Correlation windows.** For each of the three 24 hour windows, find the id range with the `Timestamp` index lookup above, read that range in chunks of `--chunk-rows` with the same paging query, resample each channel onto a 60 s UTC grid with last observation carried forward, and keep only the grid (about 1,440 points per channel). The grid is used for correlations and then discarded.
5. **Dry run.** With `--dry-run` the adapter is replaced by a recorder that prints each planned statement with its parameters (and the estimated number of rows) and executes nothing, not even the connection.

**How it works:**
- `sim/profiler/guard.py` wraps every adapter. It strips leading whitespace and comments, checks that the statement starts with `SELECT`, `SHOW`, or `SET`, rejects anything containing a `;` followed by another statement, and raises `ReadOnlyViolation` otherwise. Adapters have no method that bypasses it.
- Statistics are computed with streaming or mergeable forms where possible (counts, sums, and sums of squares for mean and std; per-chunk spacing and run-length counts merged by addition). Percentiles, medians, and MAD use the pooled sample values held as compact float32 arrays per channel (at most `chunks x chunk_rows` values, about 1 million per channel by default), never written to disk.
- Spacing, ordering, and duplicate checks only use pairs of rows inside the same chunk, because only those are truly consecutive. The 20 largest gaps keep only `(start_time, end_time, length_s)`.
- Output is built as a plain dict, checked against the allowlist schema in `sim/profiler/schema.py` (unknown key or over-long list raises), and only then written. The JSON looks like:
  ```json
  {
    "banner": "AGGREGATES ONLY. Review before sharing.",
    "server": {"version": "...", "system_time_zone": "...", "now": "...", "utc_now": "..."},
    "sampling": {"chunks": 200, "chunk_rows": 5000, "since": null, "until": null},
    "tables": {
      "Labjack": {
        "id_min": 1, "id_max": 17770893, "row_estimate": 17770893, "id_min_is_1": true,
        "id_step_1_share": 1.0, "first_ts": "...", "last_ts": "...",
        "backward_ts_share": 0.0, "duplicate_ts_share": 0.21,
        "spacing_s": {"p1": 1, "p5": 1, "p50": 2, "p95": 2, "p99": 3, "max": 4210},
        "spacing_share": {"0": 0.21, "1": 0.30, "2": 0.45, "gt2": 0.04},
        "gaps_gt_5x_median": 37, "largest_gaps": [{"start": "...", "end": "...", "length_s": 4210}],
        "channels": {
          "buffer_pressure": {"n": 1000000, "null_fraction": 0.0, "min": 0, "max": 0, "mean": 0,
            "std": 0, "median": 0, "mad_sigma": 0, "noise_mad_sigma": 0, "noise_lag1": 0,
            "slow_share": 0, "repeat_share": 0, "longest_repeat": 0}
        }
      }
    },
    "correlations": {"diff_top30": [{"a": "Labjack.x", "b": "Flow_Rates.y", "r": 0.8}], "level_top30": []}
  }
  ```
  (Numbers above are placeholders to show the shape.) `ptdb_profile.md` is rendered from the same dict, one table per feed and one row per channel.
- How I read the output: open `ptdb_profile.md` first, review every value, especially `largest_gaps` (the only individual timestamps), then decide what, if anything, to share.

**Tests (using the SQLite fixture from R2):** profiling a known simulated run recovers its baseline, sigma, and spacing within tolerance; the guard rejects every non-SELECT/SHOW/SET statement; `--dry-run` runs nothing; the output passes the allowlist check; and no raw reading from the fixture appears in the output files.

**STOP after R3.** Remind me of rule 8. Do not run the profiler against the real database. Give me the exact command to run on Shannon (with the environment variable names, a replica host if one exists, and an off-peak suggestion).

### PHASE R4: FIT THE SIMULATOR TO THE PROFILE

I will run the profiler, review the output, and then give you the profile JSON (or tell you which parts I allow).

1. `sim/from_profile.py` and `--profile path` on the scenario script set, per channel: baseline (the median), sigma (the short-term noise estimate), `ar1_phi` (the lag-1 autocorrelation), and slow wander amplitude (from the slow-component share). Per feed: the period (median spacing), jitter (from the spacing spread), and outage and gap rates (from the gap counts).
2. Couplings come only from pairs I list in a `couplings:` section of the config (chosen from the profiler's correlation lists). Never add them automatically.
3. `scripts/compare_sim_to_profile.py`: simulate with the fitted config, write it to the SQLite fixture, profile it with the same profiler, and print a table of real against simulated aggregates per channel. Flag any channel whose noise sigma or lag-1 autocorrelation differs by more than 25 percent (a starting value).
4. Document what the fit does not capture: regime changes, bursty noise, operating modes, and anything that is not an aggregate.
5. Update the README with a real-layout section, the data safety rules, how to run the profiler, and what is synthetic.
6. Tests: fitting the profile of a simulated fixture recovers the original parameters within 10 percent (a starting tolerance).

**How the data is read in R4:** the only real-data input is the profile JSON I hand over (or a trimmed copy with only the parts I allow). Nothing in R4 connects to ptdb.
- `sim/from_profile.load_profile(path)` reads the JSON and runs it through the same allowlist schema from R3, so a hand-edited or partial file is accepted only if its shape is valid. Missing tables or channels are allowed: they keep the synthetic defaults from R2 and are listed in a printed "not fitted" report.
- `fit_config(profile, registry) -> dict` returns a normal scenario config (the same format `make_layout_scenario.py` already reads), and `--profile path` simply calls it before generating. The fitted config is also written to `out/layout/<run_name>/config_used.yaml` so every fitted run can be reproduced without the profile.

**How it works:**
- Mapping from profile fields to simulator parameters:

  | Simulator parameter | Profile source |
  | --- | --- |
  | channel `baseline` | `median` |
  | channel `sigma` | `noise_mad_sigma` |
  | channel `ar1_phi` | `noise_lag1`, clipped to [0, 0.99] |
  | channel wander amplitude | from `slow_share`: `sigma * sqrt(slow_share / (1 - slow_share))`, capped at 0.99 share |
  | channel null rate | `null_fraction` |
  | feed `nominal_period_s` | `spacing_s.p50`, refined with `spacing_share` because spacing is whole seconds |
  | feed `jitter_s` | from the spread between `spacing_s.p5` and `spacing_s.p95` |
  | feed gap and outage rates | `gaps_gt_5x_median` per sampled hour, split by gap length |

- Couplings are copied only from the `couplings:` section I write. The fitter never reads `correlations` to add a coupling. It only prints them as suggestions.
- `compare_sim_to_profile.py` runs the loop end to end: fit, simulate, write the SQLite fixture (R2), profile the fixture with the same `SQLiteAdapter` and code path as the real run (R3), then print one row per channel: real value, simulated value, and percent difference for `noise_mad_sigma`, `noise_lag1`, `median`, and feed spacing. Rows over the 25 percent threshold are flagged. Because the real and simulated numbers come from the same profiler code, any mismatch is in the simulator, not in the measurement.

**STOP after R4 and report.**

---

## 6. DEFINITION OF DONE

- `pytest` passes, including all earlier tests.
- `configs/ptdb_layout.yaml` lists 109 columns and 21 included channels.
- A wide scenario can be written to a SQLite fixture with the real table and column names and read back exactly.
- `python scripts/profile_db.py --dry-run` prints the planned queries and runs none.
- No real data, credentials, or profile output is tracked in git.
- No em dashes anywhere in the repo (run a grep check and report the result).

Start by reading the existing repo (README, `sim/`, `configs/`, `tests/`). Then restate the plan in a few lines, list every `[FILL IN]` that is still empty and every assumption you are making, and begin Phase R1.
