# Data cards

Three datasets ship with the repository. The application reads none of them specially: each is loaded through the
same loader, and any folder of Annex B CSVs (file names starting with the table name) works the same way.

| Folder | Role |
|---|---|
| `data/history/` | **The past data.** Five quarters, 2025-Q3 to 2026-Q3. Loaded by default (Docker first start). The baseline learns from the 4 quarters before the current one |
| `data/demo_new_quarter/` | Example **new quarter** (2026-Q4) to upload on top of the history. Synthetic, made by `scripts/make_demo_quarter.py` |
| `data/sample/`, `data/late_drop/` | Small synthetic pack with a planted answer key, used by the tests and `scripts/run_eval.py` (card below) |

## History: factory-history-5q v1.0

| Field | Value |
|---|---|
| **Files** | `meter_15min_5q_messy.csv` 394,807 rows (main + 8 sub-meters: compressors, furnace, hvac, lighting, line_1, line_2, line_3, pumps); `tariff_5q.csv` 1,111 rows; `production_5q_messy.csv` 1,829 rows (4 product families P_A to P_D); `weather_5q_messy.csv` 456 rows. Row counts also in `data/history/data_card.json` (validation check 6). |
| **Date range** | 2025-07-01 to 2026-09-30: 2025-Q3, 2025-Q4, 2026-Q1, 2026-Q2, 2026-Q3, all complete |
| **Timestamps** | No zone written. They line up with local shift hours (lines idle 23:00 to 05:00), so the factory time zone stays UTC, meaning naive = local |
| **Tariff** | Dated time blocks: weekday off-peak 00:00-07:00 and 22:00-24:00, peak 07:00-22:00, weekends flat. Prices: peak 0.168 / off-peak 0.105 / flat 0.118, rising on 2026-05-15/16 (0.192 / 0.118 / 0.132) and 2026-08-01/03 (0.198 / 0.121 / 0.136) |
| **Shift patterns** | Clock ranges (`06:00-14:00\|14:00-22:00`, also written with a comma), `06:00-14:00` on Saturdays, `OFF` on Sundays and holidays |
| **Main meter by quarter (kWh)** | 1,434,140 / 1,361,974 / 1,406,632 / 1,531,830 / 1,864,705 |
| **Bill by quarter** | 216,084 / 207,540 / 214,472 / 249,597 / 325,113 |
| **Weather (CDD / HDD per quarter)** | 1108/0, 190/151, 18/393, 683/2, 1344/0: a full seasonal cycle, 2026-Q3 the hottest |

**Data problems found and how they are handled**

| Problem | Handling |
|---|---|
| 4 duplicate meter rows | collapsed (check 3) |
| line_2 reading of -250 kWh (reset) | replaced by interpolation (check 2) |
| 5 single missing 15-minute intervals on every meter | interpolated (check 3) |
| 3 duplicate production rows, 3 duplicate weather dates | collapsed |
| 4 missing weather dates (2025-11-07, 2026-03-19, 2026-06-02, 2026-09-11) | interpolated, left out of the baseline fit (check 5) |
| 2026-08-15 public holiday logged as a two-shift day with 0 units (lines ran at about half load) | not a working day (check 5) |
| 5 days logged with a shift but 0 units | listed in check 5; not working days |
| Unmetered load 5.3-5.7% every quarter | reported as unmetered; tolerance set to 6% (check 1) |

**Patterns visible in the data (found by the system, not planted by us):** compressor idle load steps up from about
57 kW to about 118 kW on 2026-07-10 (a smaller step on 2026-07-01, and a slow rise from about 41 to 50 kW over the
year); HVAC idle load follows both cooling and heating degree-days; furnace idle load is higher in winter; product
mix shifts strongly to P_C and P_D.

**Known limits.** The source of this data is not documented in the repository, so treat it as given. There is no
answer key, so accuracy on it is checked by consistency (additivity, the waste identity, recomputation by hand in
the tests), not against a true split.

## Demo new quarter (2026-Q4)

`python scripts/make_demo_quarter.py` builds it from 2025-Q4 shifted by 52 weeks (same weekdays): production x1.04,
compressor leak +60 kW in all hours (still running), tariff blocks at the last history prices. 79,488 meter rows,
224 tariff rows, 367 production rows, 91 weather rows. **Synthetic, not a forecast.** `--days N` writes a quarter in
progress; `--leak-fixed-on DATE` repairs the leak part-way.

---

## Synthetic sample: synthetic-factory v1.0 (Annex D)

| Field | Value |
|---|---|
| **Name and version** | synthetic-factory, v1.0, generated 11 Oct 2026 |
| **How it was made** | `python scripts/generate_data.py --seed 42`. A seeded numpy simulation; no LLM and no prompts were used to create the data. The generator was written with an AI coding assistant (see the AI-usage disclosure in the README). |
| **What it contains** | `meter_15min.csv` 158,105 rows (main + 8 sub-meters: compressors, furnace, hvac, lighting, utilities, line_1, line_2, line_3); `tariff.csv` 4 rows; `production.csv` 549 rows (3 product families x 183 days); `weather.csv` 183 rows. Date range 2026-04-01 to 2026-09-30 (2026-Q2 and 2026-Q3). Row counts are also in `data/sample/data_card.json`, which validation check 6 reads. |
| **Late-drop pack** | `data/late_drop/tariff.csv`: corrected tariff (price change moved from 1 Aug to 20 Jul, peak price 0.1456 corrected to 0.1440). `data/late_drop/meter_15min.csv`: hvac sub-meter for Q3 with a calibration fix (readings x 0.93), 8,832 rows. |
| **Privacy** | Fully synthetic. No real company or personal data. |

### Traps and edge cases planted on purpose

| Guide section 4.4 edge case | Where it is in the data |
|---|---|
| Sub-meters do not add up to the main meter | Main = sum of sub-meters + about 3% unmetered load, every interval |
| Missing intervals | main: 45 minutes on 2026-04-14 and 5 hours on 2026-07-21; hvac: 6 hours on 2026-06-03 |
| Meter resets | furnace 2026-05-20 10:00 (negative reading); line_2 2026-09-02 13:15 (register dump of 250,000) |
| Duplicates | 40 duplicated lighting rows |
| Tariff change in the middle of a quarter | Peak and off-peak prices rise 12% on 2026-08-01 |
| Mix shift that raises energy per unit without waste | Q3 makes more of product C (14 kWh per unit) and less of A (4 kWh per unit) |
| Hot period that raises cooling | 2026-08-03 to 2026-08-16 is 4 C hotter; all of Q3 is warmer than Q2 |
| Public holiday logged as a normal shift | 2026-08-26: `is_holiday` true, `shift_pattern` "2-shift", zero units, lines idle |
| Hidden waste pattern | Compressed-air leak: +35 kW on `compressors`, all hours, from 2026-08-10 |
| Standing waste (not part of the increase) | Lighting stays at 50 kW of its 80 kW when lines are stopped, in both quarters |

### Checks run

`python scripts/load_data.py data/sample --reset` runs the six validation checks. Result on this dataset: overall WARN
(checks 1, 4, 6 PASS; checks 2, 3, 5 WARN because the planted resets, gaps, duplicates and the mislogged holiday are found and handled).

### Known limits

- Energy is generated from a model close to the one the baseline assumes (linear in units and degree-days), so results on this data
  are optimistic compared with a real factory.
- One factory, two quarters, three product families, no heating season (heating degree-days are zero).
- Prices are plain numbers with no currency; action costs in `config.yaml` are assumptions in the same unit.
- Day-to-day noise is 2.5% and independent between days; real factories have drifts, start-up effects and maintenance stops.
- The answer key (`eval/truth.json`) is written by the generator and read only by `scripts/run_eval.py`.
