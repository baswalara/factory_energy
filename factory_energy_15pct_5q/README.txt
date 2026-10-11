FACTORY ENERGY COST DECOMPOSITION - SYNTHETIC DATA

Coverage: 2025-07-01 to 2026-09-30 (5 quarters)
Primary comparison: 2026-Q3 versus 2026-Q2
Main-meter electricity bill increase: EXACTLY 15.0%

Schemas:
meter_15min.csv: meter_id, ts, kwh
tariff.csv: valid_from, valid_to, period, price_per_kwh
production.csv: date, product, units, shift_pattern, is_holiday
weather.csv: date, avg_temp_c, cooling_degree_days, heating_degree_days

Designed conditions:
- 8 sub-meters + main meter
- 5 quarters
- tariff change part-way through Q3 2026: 2026-08-15
- product-mix shift to more energy-intensive products
- hotter Q3 2026 and July hot spell
- hidden compressor idle leak from 2026-07-10
- sub-meter total differs from main meter
- missing sub-meter intervals
- duplicate rows
- reset-like zero segment on one sub-meter
- public holiday logged with normal shift pattern

Q2 2026 bill: 51750.637581
Q3 2026 bill: 59513.233218
Change: 15.000000000000%

ground_truth_DO_NOT_INGEST.json is evaluation-only.
