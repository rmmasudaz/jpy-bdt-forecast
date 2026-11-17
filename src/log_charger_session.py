#!/usr/bin/env python3
"""
log_charger_session.py — Record device-only charger session samples to CSV.

Writes rows: ts, soc_pct, batt_temp_c, charge_ua, status, events
under data/charger_sessions/<charger_id>/<run_id>.csv

Also probes Android device identity via ADB (model, manufacturer, OS/version,
battery-temp availability) into device_info.json + session manifest fields.
Do not ask operators to type model/OS — collect by probe.

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
    # Probe only (writes device_info.json; no typing model/OS)
    python3 src/log_charger_session.py --probe-device

    # OEM session — auto-probes at start when --source adb
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
DEVICE_INFO_NAME = "device_info.json"


class ProbeError(RuntimeError):
    """ADB device probe failed; do not invent model/OS (fail closed)."""


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


def adb_argv(adb_bin: str, serial: str | None, *shell_args: str) -> list[str]:
    cmd = [adb_bin]
    if serial:
        cmd += ["-s", serial]
    cmd.extend(shell_args)
    return cmd


def adb_run(adb_bin: str, serial: str | None, *shell_args: str, timeout: float = 30) -> str:
    cmd = adb_argv(adb_bin, serial, *shell_args)
    try:
        proc = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as e:
        raise ProbeError(
            "adb not found on PATH. Install Android platform-tools, enable USB debugging, "
            "then re-run (e.g. --probe-device or --source adb)."
        ) from e
    except subprocess.CalledProcessError as e:
        err = (e.stderr or e.stdout or str(e)).strip()
        raise ProbeError(f"adb command failed ({' '.join(cmd)}): {err}") from e
    except subprocess.TimeoutExpired as e:
        raise ProbeError(f"adb command timed out ({' '.join(cmd)})") from e
    return proc.stdout


def adb_getprop(adb_bin: str, serial: str | None, key: str) -> str | None:
    val = adb_run(adb_bin, serial, "shell", "getprop", key).strip()
    # getprop prints a blank line when the property is unset.
    return val if val else None


def _parse_dumpsys_battery_fields(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if ":" not in line:
            continue
        key, _, val = line.partition(":")
        fields[key.strip().lower()] = val.strip()
    return fields


def _temp_c_from_fields(fields: dict[str, str]) -> float | None:
    temp_raw = fields.get("temperature")
    if temp_raw in (None, ""):
        return None
    try:
        # Android dumpsys temperature is tenths of a degree Celsius.
        return round(float(temp_raw) / 10.0, 1)
    except ValueError:
        return None


def _charge_ua_from_fields(fields: dict[str, str]) -> int | None:
    current = None
    for key in ("current now", "current_now", "max charging current", "charger current"):
        if key in fields:
            current = fields[key]
            break
    if current in (None, ""):
        return None
    try:
        m = re.search(r"-?\d+", current.replace(",", ""))
        if m:
            return int(m.group(0))
    except ValueError:
        return None
    return None


def _status_from_fields(fields: dict[str, str]) -> str:
    status_code = fields.get("status")
    status = STATUS_MAP.get(status_code or "", None)
    if status is None:
        word = (fields.get("status") or "unknown").lower().replace(" ", "_")
        status = word if word in STATUS_MAP.values() else "unknown"
    return status


def _parse_dumpsys_battery(text: str) -> dict[str, Any]:
    """Parse `adb shell dumpsys battery` into a sample dict (nullable fields OK)."""
    fields = _parse_dumpsys_battery_fields(text)
    soc = fields.get("level")
    if soc is None:
        raise RuntimeError("dumpsys battery did not include level/SoC")
    return {
        "ts": utc_now(),
        "soc_pct": float(soc),
        "batt_temp_c": _temp_c_from_fields(fields),
        "charge_ua": _charge_ua_from_fields(fields),
        "status": _status_from_fields(fields),
        "events": "",
    }


def sample_adb(adb_bin: str, serial: str | None) -> dict[str, Any]:
    try:
        text = adb_run(adb_bin, serial, "shell", "dumpsys", "battery")
    except ProbeError as e:
        raise SystemExit(str(e)) from e
    return _parse_dumpsys_battery(text)


def probe_android_device(adb_bin: str, serial: str | None) -> dict[str, Any]:
    """
    Collect model / manufacturer / OS / battery-temp availability via ADB only.
    Never invents values — missing props stay null (fail closed).
    """
    manufacturer = adb_getprop(adb_bin, serial, "ro.product.manufacturer")
    model = adb_getprop(adb_bin, serial, "ro.product.model")
    product_device = adb_getprop(adb_bin, serial, "ro.product.device")
    brand = adb_getprop(adb_bin, serial, "ro.product.brand")
    os_version = adb_getprop(adb_bin, serial, "ro.build.version.release")
    sdk_raw = adb_getprop(adb_bin, serial, "ro.build.version.sdk")
    build_id = adb_getprop(adb_bin, serial, "ro.build.display.id")

    sdk_int: int | None = None
    if sdk_raw is not None:
        try:
            sdk_int = int(sdk_raw)
        except ValueError:
            sdk_int = None

    resolved_serial = serial
    if not resolved_serial:
        try:
            resolved_serial = adb_run(adb_bin, None, "get-serialno").strip() or None
            if resolved_serial in ("unknown", "no devices/emulators found"):
                resolved_serial = None
        except ProbeError:
            resolved_serial = None

    batt_text = adb_run(adb_bin, serial, "shell", "dumpsys", "battery")
    batt_fields = _parse_dumpsys_battery_fields(batt_text)
    batt_temp_c = _temp_c_from_fields(batt_fields)
    charge_ua = _charge_ua_from_fields(batt_fields)
    # Temp is "available" only if dumpsys exposes a parseable temperature key.
    batt_temp_available = "temperature" in batt_fields and batt_temp_c is not None
    charge_ua_available = charge_ua is not None

    return {
        "schema_version": 1,
        "probed_at": utc_now(),
        "probe_source": "adb",
        "platform": "android",
        "manufacturer": manufacturer,
        "brand": brand,
        "model": model,
        "product_device": product_device,
        "os_name": "Android",
        "os_version": os_version,
        "sdk_int": sdk_int,
        "build_id": build_id,
        "serial": resolved_serial,
        "batt_temp_available": batt_temp_available,
        "batt_temp_c_sample": batt_temp_c,
        "charge_ua_available": charge_ua_available,
        "charge_ua_sample": charge_ua,
        "evidence_constraint": "device_logs_and_manual_notes_only",
        "no_extra_devices": True,
        "notes": (
            "Collected by ADB getprop + dumpsys battery. "
            "If batt_temp_available is false, heat evidence is weaker — say so before any verdict."
        ),
    }


def write_device_info(path: Path, info: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")


def device_fields_for_manifest(info: dict[str, Any]) -> dict[str, Any]:
    """Compact session fields derived from a probe result (no secrets)."""
    model = info.get("model")
    manufacturer = info.get("manufacturer")
    os_version = info.get("os_version")
    os_name = info.get("os_name") or "Android"
    device_os = f"{os_name} {os_version}".strip() if os_version else os_name
    return {
        "device_model": model,
        "device_manufacturer": manufacturer,
        "device_os": device_os,
        "device_sdk_int": info.get("sdk_int"),
        "device_serial": info.get("serial"),
        "batt_temp_available": info.get("batt_temp_available"),
        "device_info": DEVICE_INFO_NAME,
        "device_probed_at": info.get("probed_at"),
    }


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
    device_info: dict[str, Any] | None = None,
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
    entry: dict[str, Any] = {
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
    if device_info:
        entry.update(device_fields_for_manifest(device_info))
        data["device"] = {
            "manufacturer": device_info.get("manufacturer"),
            "model": device_info.get("model"),
            "os": device_fields_for_manifest(device_info)["device_os"],
            "batt_temp_available": device_info.get("batt_temp_available"),
            "device_info": DEVICE_INFO_NAME,
            "probed_at": device_info.get("probed_at"),
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
    p = argparse.ArgumentParser(
        description="Log a charger comparison session (device-only signals) and/or probe Android device info via ADB."
    )
    p.add_argument(
        "--probe-device",
        action="store_true",
        help="Probe Android model/OS/batt-temp via ADB and write device_info.json (also implied at adb session start).",
    )
    p.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip auto device probe at session start (ADB sessions probe by default).",
    )
    p.add_argument("--charger-id", choices=VALID_CHARGERS, default=None)
    p.add_argument("--run-id", default=None, help="Unique run id, e.g. oem_r01")
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
    args = p.parse_args(argv)

    probe_only = args.probe_device and args.charger_id is None and args.run_id is None
    if probe_only:
        return args
    if args.charger_id is None or args.run_id is None:
        p.error("--charger-id and --run-id are required unless running probe-only (--probe-device with no session ids)")
    return args


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Run ADB probe and write device_info.json. Raises ProbeError on failure."""
    if shutil.which(args.adb.split()[0]) is None and args.adb == "adb":
        print(
            "warning: adb not found on PATH — install platform-tools or pass --adb /path/to/adb",
            file=sys.stderr,
        )
    info = probe_android_device(args.adb, args.serial)
    out = args.out_root / DEVICE_INFO_NAME
    write_device_info(out, info)
    print(f"device probe -> {out}", file=sys.stderr)
    print(
        f"  {info.get('manufacturer')} {info.get('model')} | "
        f"{info.get('os_name')} {info.get('os_version')} (sdk {info.get('sdk_int')}) | "
        f"batt_temp_available={info.get('batt_temp_available')}",
        file=sys.stderr,
    )
    return info


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    probe_only = args.probe_device and args.charger_id is None and args.run_id is None

    # Probe-only path: one command, no session logging.
    if probe_only:
        try:
            run_probe(args)
        except ProbeError as e:
            print(f"device probe failed (fail closed): {e}", file=sys.stderr)
            return 1
        return 0

    assert args.charger_id is not None and args.run_id is not None
    out_csv = args.out_root / args.charger_id / f"{args.run_id}.csv"
    csv_rel = f"{args.charger_id}/{args.run_id}.csv"
    manifest_path = args.out_root / "manifest.json"

    device_info: dict[str, Any] | None = None
    # Auto-probe at ADB session start (or when --probe-device). Never invent identity.
    should_probe = (args.probe_device or args.source == "adb") and not args.no_probe
    if should_probe:
        try:
            device_info = run_probe(args)
        except ProbeError as e:
            print(f"device probe failed (fail closed): {e}", file=sys.stderr)
            return 1

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
                        device_info=device_info,
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
