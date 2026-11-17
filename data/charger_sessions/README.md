# Charger session data

Device-only charge logs for OEM vs add-on comparison.

**Constraint:** no USB power meter, IR thermometer, or other lab gear.
Evidence = phone/device logs + manual notes only.

## Layout

```
data/charger_sessions/
├── README.md                 # this file
├── manifest.json             # session metadata (labels, cable, port, ambient, SoC window, repeats)
├── oem/                      # OEM baseline CSVs: <run_id>.csv
├── addon_a/                  # add-on A CSVs
└── addon_b/                  # add-on B CSVs
```

## CSV columns

| Column | Type | Notes |
| --- | --- | --- |
| `ts` | ISO-8601 UTC | Sample timestamp |
| `soc_pct` | float | Battery state of charge (%) |
| `batt_temp_c` | float or empty | Device thermistor °C if OS exposes it; else blank |
| `charge_ua` | int or empty | Instantaneous charge current (µA) if exposed; else blank |
| `status` | string | e.g. `charging`, `not_charging`, `full`, `discharging`, `unknown` |
| `events` | string | Semicolon-separated tags (e.g. `start`; `thermal_warning`; blank if none) |

Header row required. Nullable fields are left empty — do not invent values.

## Manifest fields

See `manifest.json`. Each session entry should include at least:

- `charger_id`: `oem` | `addon_a` | `addon_b`
- `run_id`, `repeat_index`
- `cable`, `port`, `ambient`
- `soc_start_pct`, `soc_end_target_pct` (default window 20–25 → 80)
- `csv` path relative to this directory
- `manual_notes` (absolute-stop observations with timestamps)

## Safety

Safety is a **hard gate**. Do not claim Better / Equivalent / Worse without a
recorded `safety_pass` derived from device logs + absolute-stop notes.
Unavailable electrical or surface-temp instrumentation does **not** relax the gate.

## Logging

```bash
# Android (ADB) — see repo README / CONTRIBUTING for full OEM baseline steps
python3 src/log_charger_session.py \
  --charger-id oem --run-id oem_r01 --repeat-index 1 \
  --source adb --interval 5
```
