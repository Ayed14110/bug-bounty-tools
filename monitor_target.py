#!/usr/bin/env python3
"""
monitor_target.py - re-run ayed_recon on a target and report whether the
key findings changed since the last run. Prints a one-line verdict:
    CHANGED: <summary of diffs>
    UNCHANGED
State is kept in reports/.monitor_state_<host>.json
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))


def key_findings(data):
    fp = data.get("http", {})
    return {
        "status": fp.get("status_code"),
        "final_url": fp.get("final_url"),
        "technologies": sorted(
            f"{t['product']} {t.get('version') or ''}".strip()
            for t in fp.get("technologies", [])),
        "missing_headers": sorted(
            fp.get("security_headers", {}).get("missing", [])),
        "open_ports": sorted(p["port"] for p in data.get("ports", {}).get("open", [])),
        "os_guess": data.get("os", {}).get("guess"),
        "owasp": sorted(f"{c}:{n}" for c, n, _ in data.get("owasp_flags", [])),
    }


def diff(old, new):
    changes = []
    for k in new:
        if old.get(k) != new.get(k):
            changes.append(f"{k}: {old.get(k)} -> {new.get(k)}")
    return changes


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "testaspnet.vulnweb.com"
    host = re.sub(r"^https?://", "", target).split("/")[0]
    # per-cycle scans go into a gitignored subfolder to avoid repo churn
    reports = os.path.join(HERE, "reports", "monitor")
    os.makedirs(reports, exist_ok=True)
    state_path = os.path.join(reports, f".monitor_state_{host}.json")

    # run scan (skip live CVE to keep each cycle fast)
    subprocess.run([sys.executable, os.path.join(HERE, "ayed_recon.py"),
                    target, "--no-cve", "--outdir", reports],
                   capture_output=True, text=True, timeout=150)

    # newest json report for this host
    jsons = sorted(f for f in os.listdir(reports)
                   if f.startswith(f"recon_{host}") and f.endswith(".json"))
    if not jsons:
        print("ERROR: no report produced")
        return
    with open(os.path.join(reports, jsons[-1])) as f:
        data = json.load(f)
    new = key_findings(data)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    if os.path.exists(state_path):
        with open(state_path) as f:
            old = json.load(f)
        changes = diff(old, new)
        if changes:
            print(f"CHANGED @ {ts}: " + " | ".join(changes))
        else:
            print(f"UNCHANGED @ {ts}")
    else:
        print(f"BASELINE @ {ts}: {json.dumps(new)}")

    with open(state_path, "w") as f:
        json.dump(new, f, indent=2)


if __name__ == "__main__":
    main()
