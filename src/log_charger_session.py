#!/usr/bin/env python3
"""
log_charger_session.py — Record device-only charger session samples to CSV.

Writes rows: ts, soc_pct, batt_temp_c, charge_ua, status, events
under data/charger_sessions/<charger_id>/<run_id>.csv

Evidence constraint: device logs + manual notes only.
No USB meter / IR / lab gear paths.

Sources:
  adb   — Android via `adb shell dumpsys battery` (requires adb + USB debugging)
  demo  — synthetic samples for schema dry-runs (not real evidence)
  stdin — one JSON object per line from an on-device or custom sampler

iOS: there is no ADB equivalent. Public iOS APIs typically do not expose battery
temperature or charge current to third-party apps. Prefer Android for P1 logging,
or export Console/Shortcuts SoC+status only and leave batt_temp_c / charge_ua blank
(weaker heat evidence — document in the manifest).

Usage:
    python3 src/log_charger_session.py \\
      --charger-id oem --run-id oem_r01 --repeat-index 1 \\
      --source adb --interval 5

    python3 src/log_charger_session.py --charger-id oem --run-id dry --source demo --samples 3
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "charger_sessions"
VALID_CHARGERS = ("oem", "addon_a", "addon_b")

STATUS_MAP = {
    "1": "unknown",
    "2": "charging",
    "3": "discharging",
    "4": "not_charging",
    "5": "full",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ts", "soc_pct", "batt_temp_c", "charge_ua", "status", "events"])


def append_row(path: Path, row: dict[str, Any]) -> None:
    ensure_csv(path)
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                row["ts"],
                row["soc_pct"],
                "" if row.get("batt_temp_c") is None else row["batt_temp_c"],
                "" if row.get("charge_ua") is None else row["charge_ua"],
                row.get("status") or "unknown",
                row.get("events") or "",
            ]
        )


def _parse_dumpsys_battery(text: str) -> dict[str, Any]:
    """Parse `adb shell dumpsys battery` into a sample dict (nullable fields OK)."""
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        fields[key.strip().lower()] = val.strip()

    soc = fields.get("level")
    temp_raw = fields.get("temperature")
    status_code = fields.get("status")
    # Prefer instantaneous current when present (µA). Names vary by OEM/Android version.
    current = None
    for key in ("current now", "current_now", "max charging current", "charger current"):
        if key in fields:
            current = fields[key]
            break
    # Sometimes appears as "USB power: ..." — leave null if not a plain integer.

    batt_temp_c = None
    if temp_raw not in (None, ""):
        try:
            # Android dumpsys temperature is tenths of a degree Celsius.
            batt_temp_c = round(float(temp_raw) / 10.0, 1)
        except ValueError:
            batt_temp_c = None

    charge_ua = None
    if current not in (None, ""):
        try:
            # Strip units if present (e.g. "1200000", "-500000 uA").
            m = re.search(r"-?\d+", current.replace(",", ""))
            if m:
                charge_ua = int(m.group(0))
        except ValueError:
            charge_ua = None

    status = STATUS_MAP.get(status_code or "", None)
    if status is None:
        # Some devices print words instead of codes.
        word = (fields.get("status") or "unknown").lower().replace(" ", "_")
        status = word if word in STATUS_MAP.values() else "unknown"

    if soc is None:
        raise RuntimeError("dumpsys battery did not include level/SoC")

    return {
        "ts": utc_now(),
        "soc_pct": float(soc),
        "batt_temp_c": batt_temp_c,
        "charge_ua": charge_ua,
        "status": status,
        "events": "",
    }


def sample_adb(adb_bin: str, serial: str | None) -> dict[str, Any]:
    cmd = [adb_bin]
    if serial:
        cmd += ["-s", serial]
    cmd += ["shell", "dumpsys", "battery"]
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as e:
        raise SystemExit(
            "adb not found on PATH. Install Android platform-tools, enable USB debugging, "
            "then re-run with --source adb."
        ) from e
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"adb dumpsys battery failed: {e.stderr or e}") from e
    except subprocess.TimeoutExpired as e:
        raise SystemExit("adb dumpsys battery timed out") from e
    return _parse_dumpsys_battery(proc.stdout)


def sample_demo(i: int) -> dict[str, Any]:
    """Synthetic row for pipeline dry-runs — not valid comparison evidence."""
    return {
        "ts": utc_now(),
        "soc_pct": 20.0 + i,
        "batt_temp_c": 30.0 + 0.1 * i,
        "charge_ua": 1_500_000,
        "status": "charging",
        "events": "demo" if i == 0 else "",
    }


def sample_stdin(line: str) -> dict[str, Any]:
    """
    Accept one JSON object per line. Required: soc_pct.
    Optional: ts, batt_temp_c, charge_ua, status, events.
    Missing optional numerics stay null (fail closed — do not invent).
    """
    obj = json.loads(line)
    if "soc_pct" not in obj:
        raise ValueError("stdin JSON sample requires soc_pct")
    return {
        "ts": obj.get("ts") or utc_now(),
        "soc_pct": float(obj["soc_pct"]),
        "batt_temp_c": None if obj.get("batt_temp_c") in (None, "") else float(obj["batt_temp_c"]),
        "charge_ua": None if obj.get("charge_ua") in (None, "") else int(obj["charge_ua"]),
        "status": obj.get("status") or "unknown",
        "events": obj.get("events") or "",
    }


def update_manifest_stub(
    manifest_path: Path,
    *,
    charger_id: str,
    run_id: str,
    repeat_index: int,
    csv_rel: str,
    cable: str,
    port: str,
    ambient: str,
    soc_start: float | None,
    soc_end: float,
) -> None:
    """Append or refresh a session entry; never sets safety_pass."""
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        data = {
            "schema_version": 1,
            "evidence_constraint": "device_logs_and_manual_notes_only",
            "no_extra_devices": True,
            "safety_gate": "hard",
            "chargers": {},
            "sessions": [],
        }

    sessions = data.setdefault("sessions", [])
    entry = {
        "run_id": run_id,
        "charger_id": charger_id,
        "repeat_index": repeat_index,
        "cable": cable,
        "port": port,
        "ambient": ambient,
        "soc_start_pct": soc_start,
        "soc_end_target_pct": soc_end,
        "csv": csv_rel,
        "started_at": utc_now(),
        "ended_at": None,
        "status": "logging",
        "manual_notes": [],
    }
    # Replace same run_id if re-logging; else append.
    replaced = False
    for i, s in enumerate(sessions):
        if s.get("run_id") == run_id:
            sessions[i] = {**s, **entry}
            replaced = True
            break
    if not replaced:
        sessions.append(entry)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def mark_manifest_ended(manifest_path: Path, run_id: str) -> None:
    if not manifest_path.exists():
        return
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    for s in data.get("sessions", []):
        if s.get("run_id") == run_id:
            s["ended_at"] = utc_now()
            if s.get("status") == "logging":
                s["status"] = "recorded"
            break
    manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log a charger comparison session (device-only signals).")
    p.add_argument("--charger-id", required=True, choices=VALID_CHARGERS)
    p.add_argument("--run-id", required=True, help="Unique run id, e.g. oem_r01")
    p.add_argument("--repeat-index", type=int, default=1)
    p.add_argument("--source", choices=("adb", "demo", "stdin"), default="adb")
    p.add_argument("--interval", type=float, default=5.0, help="Seconds between samples (adb/demo)")
    p.add_argument("--samples", type=int, default=0, help="Stop after N samples (0 = until Ctrl-C / stdin EOF)")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--adb", default=os.environ.get("ADB", "adb"), help="adb binary (or set ADB=)")
    p.add_argument("--serial", default=None, help="adb device serial if multiple devices")
    p.add_argument("--cable", default="unlabeled")
    p.add_argument("--port", default="device_usb_c")
    p.add_argument("--ambient", default="unnoted")
    p.add_argument("--soc-end-target", type=float, default=80.0)
    p.add_argument("--no-manifest", action="store_true", help="Do not touch manifest.json")
    p.add_argument(
        "--event",
        action="append",
        default=[],
        help="Tag to attach to the first sample only (repeatable), e.g. --event start",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_csv = args.out_root / args.charger_id / f"{args.run_id}.csv"
    csv_rel = f"{args.charger_id}/{args.run_id}.csv"
    manifest_path = args.out_root / "manifest.json"

    if args.source == "adb" and shutil.which(args.adb.split()[0]) is None and args.adb == "adb":
        print(
            "warning: adb not found on PATH — install platform-tools or pass --adb /path/to/adb",
            file=sys.stderr,
        )

    print(f"logging -> {out_csv}", file=sys.stderr)
    print(
        "safety remains a hard gate; this script only records samples "
        "(null temp/current stay empty — fail closed).",
        file=sys.stderr,
    )

    n = 0
    first_soc: float | None = None
    try:
        while True:
            if args.source == "adb":
                row = sample_adb(args.adb, args.serial)
            elif args.source == "demo":
                row = sample_demo(n)
            else:
                line = sys.stdin.readline()
                if not line:
                    break
                line = line.strip()
                if not line:
                    continue
                row = sample_stdin(line)

            if n == 0 and args.event:
                existing = row.get("events") or ""
                tags = [t for t in args.event if t]
                row["events"] = ";".join([existing, *tags] if existing else tags)

            if first_soc is None:
                first_soc = float(row["soc_pct"])
                if not args.no_manifest:
                    update_manifest_stub(
                        manifest_path,
                        charger_id=args.charger_id,
                        run_id=args.run_id,
                        repeat_index=args.repeat_index,
                        csv_rel=csv_rel,
                        cable=args.cable,
                        port=args.port,
                        ambient=args.ambient,
                        soc_start=first_soc,
                        soc_end=args.soc_end_target,
                    )

            append_row(out_csv, row)
            n += 1
            print(
                f"{row['ts']}  soc={row['soc_pct']}  temp={row.get('batt_temp_c')}  "
                f"ua={row.get('charge_ua')}  status={row.get('status')}",
                file=sys.stderr,
            )

            if args.samples and n >= args.samples:
                break
            if args.source != "stdin":
                time.sleep(max(args.interval, 0.2))
    except KeyboardInterrupt:
        print("\nstopped by user", file=sys.stderr)
    finally:
        if not args.no_manifest and first_soc is not None:
            mark_manifest_ended(manifest_path, args.run_id)

    print(f"wrote {n} sample(s) to {out_csv}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
