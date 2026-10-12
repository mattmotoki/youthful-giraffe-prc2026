"""Deterministic schema and internal-consistency checks for exp-082 inventory.json.

Reads only experiments/exp-082-scoremap/inventory.json and checks that every cited repo path
exists. It opens no data file. Exit 0 when every check passes; otherwise it prints the failures
and exits 1. The receipt goes to analysis/check-receipt.json (refuses to overwrite a different
receipt unless --rewrite is given, because the receipt is regenerated with the inventory).
"""

import argparse
import hashlib
import json
import pathlib
import re
import sys

EXP = pathlib.Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]

STATUS = {
    "served", "open-executable", "open-blocked", "maybe", "closed-provisional",
    "closed-measured", "refuted", "incomplete", "unregistered", "rule-blocked", "not-run",
}
EVIDENCE = {"measured", "measured-selected", "projected", "oracle-bound", "incomplete", "none"}
BLOCKERS = {
    "matt-public-action", "private-credential", "paid-data", "new-rule", "missing-exact-bytes",
    "unavailable-machine", "missing-observation", "failed-prerequisite", "none",
}
REQUIRED = [
    "id", "name", "family", "pro_queue", "population", "allowed_inputs", "target_or_contrast",
    "destination", "reachable_estimate", "uncertainty", "transfer_evidence", "status",
    "evidence_quality", "failed_repairs", "revival_condition", "compute_cost", "resource_risk",
    "input_rule_risk", "contamination_selection", "blockers", "shares_destination_with",
    "needs_observation", "files",
]
PRO_IDS = [f"Q{n:02d}" for n in range(3, 17)]
CITE = re.compile(r"^([A-Za-z0-9_./-]+?)(?::(\d+)(?:-(\d+))?)?(?:\s*(?:§|#).*)?$")


def cite_path_ok(cite):
    m = CITE.match(cite.strip())
    if not m:
        return False, "unparsable"
    path = REPO / m.group(1)
    if not path.exists():
        return False, "missing path"
    if m.group(2) and path.is_file():
        n_lines = sum(1 for _ in path.open("rb"))
        hi = int(m.group(3) or m.group(2))
        if hi > n_lines:
            return False, f"line {hi} beyond {n_lines}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rewrite", action="store_true")
    args = ap.parse_args()

    inv_path = EXP / "inventory.json"
    raw = inv_path.read_bytes()
    inv = json.loads(raw)
    fails, checks = [], 0

    def check(cond, msg):
        nonlocal checks
        checks += 1
        if not cond:
            fails.append(msg)

    mech = inv["mechanisms"]
    ids = [m.get("id") for m in mech]
    check(len(ids) == len(set(ids)), "duplicate mechanism ids")
    by_id = {m["id"]: m for m in mech}

    n_cites = 0
    for m in mech:
        mid = m.get("id", "?")
        for k in REQUIRED:
            check(k in m, f"{mid}: missing field {k}")
        if any(k not in m for k in REQUIRED):
            continue
        check(m["status"] in STATUS, f"{mid}: bad status {m['status']}")
        check(m["evidence_quality"] in EVIDENCE, f"{mid}: bad evidence {m['evidence_quality']}")
        check(isinstance(m["blockers"], list) and m["blockers"], f"{mid}: blockers must be a list")
        for b in m["blockers"]:
            check(b in BLOCKERS, f"{mid}: bad blocker {b}")
        check(not ("none" in m["blockers"] and len(m["blockers"]) > 1), f"{mid}: none with blockers")
        blocked = m["blockers"] != ["none"]
        if m["status"] == "open-executable":
            check(not blocked, f"{mid}: open-executable but blocked {m['blockers']}")
        if m["status"] in {"open-blocked", "rule-blocked"}:
            check(blocked, f"{mid}: {m['status']} without a blocker")
        for key in ("destination", "reachable_estimate"):
            v = m[key]
            check(isinstance(v, dict) and {"value", "unit", "basis"} <= set(v), f"{mid}: {key} shape")
        d, r = m["destination"], m["reachable_estimate"]
        if (isinstance(d.get("value"), (int, float)) and isinstance(r.get("high"), (int, float))
                and d.get("unit") == r.get("unit")):
            check(r["high"] <= d["value"] + 1e-9, f"{mid}: reachable high above destination")
        for other in m["shares_destination_with"]:
            check(other in by_id, f"{mid}: shares_destination_with unknown id {other}")
            if other in by_id:
                check(mid in by_id[other]["shares_destination_with"],
                      f"{mid}: overlap with {other} not symmetric")
        check(isinstance(m["files"], list) and m["files"], f"{mid}: no supporting files")
        for c in m["files"]:
            ok, why = cite_path_ok(c)
            n_cites += 1
            check(ok, f"{mid}: cite {c!r} {why}")

    covered = {q for m in mech for q in (m["pro_queue"] or [])}
    for q in PRO_IDS:
        check(q in covered, f"Pro queue item {q} not covered")

    for row in inv["return_table"]:
        check(row["mechanism_id"] in by_id, f"return-table row {row['row']} maps to unknown id")
        ok, why = cite_path_ok(row["cite"])
        check(ok, f"return-table row {row['row']} cite {why}")
    check(len(inv["return_table"]) == inv["meta"]["return_table_rows_expected"],
          "return-table row count differs from DECISIONS.md")

    ranking = inv["ranking"]
    for pos, entry in enumerate(ranking, 1):
        m = by_id.get(entry["mechanism_id"])
        check(m is not None, f"ranking {pos}: unknown id")
        if m is None:
            continue
        if pos == 1:
            check(m["blockers"] == ["none"], "rank 1 is blocked")
            check(m["status"] == "open-executable", "rank 1 is not open-executable")
    for pkt in inv["packets"]:
        check(pkt["mechanism_id"] in by_id, f"packet {pkt['id']}: unknown mechanism")
        if pkt["mechanism_id"] in by_id:
            check(by_id[pkt["mechanism_id"]]["blockers"] == ["none"], f"packet {pkt['id']}: blocked")
        for col in pkt["collides_with"]:
            check(col["mechanism_id"] in by_id, f"packet {pkt['id']}: collision id unknown")
    check(len(inv["packets"]) >= 3, "fewer than three packets")
    rec = [p for p in inv["packets"] if p.get("role") == "recommended"]
    check(len(rec) == 1, "exactly one recommended packet required")

    census = inv["census"]
    shares = [p["share_board_sq_err_pct"] for p in census["partition"]]
    check(abs(sum(shares) - 100.0) <= census.get("share_tolerance_pct", 0.05),
          f"census partition shares sum to {sum(shares):.3f}")
    for p in census["partition"]:
        ok, why = cite_path_ok(p["cite"])
        check(ok, f"census {p['name']} cite {why}")

    for t in inv["top_destinations"]:
        for mid in t["mechanism_ids"]:
            check(mid in by_id, f"top destination {t['rank']}: unknown id {mid}")
    check(len(inv["top_destinations"]) == 5, "top_destinations must have five rows")
    for s in inv["stop_list"]:
        check(s["mechanism_id"] in by_id, f"stop list: unknown id {s['mechanism_id']}")
        if s["mechanism_id"] in by_id:
            check(by_id[s["mechanism_id"]]["status"] != "open-executable",
                  f"stop list holds open-executable {s['mechanism_id']}")

    receipt = {
        "inventory_sha256": hashlib.sha256(raw).hexdigest(),
        "mechanisms": len(mech),
        "citations_checked": n_cites,
        "checks": checks,
        "failures": fails,
        "pass": not fails,
    }
    out = EXP / "analysis" / "check-receipt.json"
    out.parent.mkdir(exist_ok=True)
    body = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if out.exists() and out.read_text() != body and not args.rewrite:
        print("receipt differs; rerun with --rewrite after an inventory change", file=sys.stderr)
        sys.exit(2)
    out.write_text(body)
    print(json.dumps({k: receipt[k] for k in ("mechanisms", "citations_checked", "checks", "pass")}))
    for f in fails:
        print("FAIL", f)
    sys.exit(0 if not fails else 1)


if __name__ == "__main__":
    main()
