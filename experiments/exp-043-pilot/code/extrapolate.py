"""exp-043-pilot step 4: what the pilot's numbers would be worth, and what they cost.

Everything here is arithmetic on numbers measured elsewhere in this experiment plus
two facts from the record: v10's board score and the ranking file's airport mix.
Nothing is fitted.  Every step that leaves the measured ground is marked.

  .venv/bin/python experiments/exp-043-pilot/code/extrapolate.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]

BOARD_V10 = 263.0464          # docs/pro-reasoning-2026-09-08/QUEUE.md, 8 Sep 04:49 UTC
BOARD_LEAD = 260.9310
RANK_ROWS = 344841
EHAM_ROWS = 38182             # ranking.parquet, ADEP_mvt == EHAM, PHASE_mvt == DEP
EHAM_NONM = 1206              # exp-042 REPORT.md 2.2
GB_PER_MONTH = {"2026-01": 91.7, "2026-07": 121.5, "2025-07": 108.6, "2025-all": 1106.8}


def main() -> None:
    match = json.loads((EXP / "analysis/match-heldout.json").read_text())
    al = {k: json.loads((EXP / f"analysis/align-{k}-heldout.json").read_text())
          for k in ("obs_offblock_loose", "obs_offblock_disp")}
    run = [json.loads(p.read_text()) for p in sorted((EXP / "analysis").glob("run_2025-07-*.json"))
           if json.loads(p.read_text())["probe_bytes"] == 0]

    reach = match["reach"]
    nm_share = 1 - EHAM_NONM / EHAM_ROWS
    # reach on the ranking file's EHAM rows, at the measured per-group rates
    r_nm = reach["matched_nm"]
    r_no = reach["matched_no_nm"] or 0.0
    off_share = reach["with_obs_offblock_disp"] / max(reach["matched"], 1)
    eham_reach = (nm_share * r_nm + (1 - nm_share) * r_no) * off_share
    covered = EHAM_ROWS * eham_reach

    mse_v10 = BOARD_V10 ** 2
    out = {
        "measured": {
            "days": match["days"],
            "eham_departures": reach["departures"],
            "matched_share": reach["matched_share"],
            "matched_nm": r_nm, "matched_no_nm": r_no,
            "obs_offblock_share_of_matched": round(off_share, 4),
        },
        "ranking_file": {
            "departures": RANK_ROWS, "eham": EHAM_ROWS,
            "eham_share": round(EHAM_ROWS / RANK_ROWS, 4),
            "eham_no_nm": EHAM_NONM,
            "projected_eham_reach": round(eham_reach, 4),
            "projected_covered_rows": int(covered),
            "covered_share_of_ranking": round(covered / RANK_ROWS, 4),
        },
        "board": {"v10": BOARD_V10, "lead": BOARD_LEAD,
                  "gap_s": round(BOARD_V10 - BOARD_LEAD, 4),
                  "v10_pooled_mse_s2": round(mse_v10, 1)},
        "if_eham_only": {}, "if_every_airport_matched_eham": {},
    }
    for k, a in al.items():
        # the headline gain is the one whose offset AND amplitude were frozen on
        # the development days; the cross-fit figure is carried beside it
        dev = a.get("frozen_on_development_days")
        g = (dev or a["leave_one_day_out_scaled"])["full_amplitude_mse_gain_s2"]
        out.setdefault("gain_source", {})[k] = {
            "frozen_on_development_days_s2": dev and dev["full_amplitude_mse_gain_s2"],
            "leave_one_day_out_crossfit_s2":
                a["leave_one_day_out_scaled"]["full_amplitude_mse_gain_s2"],
            "used": "frozen_on_development_days" if dev else "leave_one_day_out_scaled"}
        for name, rows in (("if_eham_only", covered),
                           ("if_every_airport_matched_eham", RANK_ROWS * eham_reach)):
            drop = g * rows / RANK_ROWS
            out[name][k] = {
                "per_matched_row_mse_gain_s2": round(g, 1),
                "pooled_mse_reduction_s2": round(drop, 1),
                "board_rmse_after_s": round((mse_v10 - drop) ** 0.5, 4),
                "board_gain_s": round(BOARD_V10 - (mse_v10 - drop) ** 0.5, 4),
                "share_of_gap_to_lead": round(
                    (BOARD_V10 - (mse_v10 - drop) ** 0.5) / (BOARD_V10 - BOARD_LEAD), 3),
            }
    sec = sum(r["seconds"] for r in run) / max(len(run), 1)
    # what the pilot actually pulled over the wire is the PUBLISHED asset size.  The
    # sum of tar member payloads is 382.3 MB smaller, because it leaves out the tar
    # headers and the padding to the 512-byte block, and reporting it as the transfer
    # understated the cost by 1.6 % (round 3).
    pub = sum(sum(a["size"] for a in r["assets_published"]) for r in run)
    raw = sum(r["raw_member_bytes"] for r in run)
    gb = pub / 1e9 / max(len(run), 1)
    out["cost"] = {
        "measured_seconds_per_day_kernel": round(sec, 1),
        "measured_gb_per_day": round(gb, 2),
        "gb_per_day_definition": "published release asset bytes, the bytes transferred",
        "note_archive_is_global": ("one pass over a day's archive yields EVERY airport, "
                                   "so the cost is per MONTH, not per airport-month: ten "
                                   "boxes cost the same transfer as one"),
        "ranking_months_gb": GB_PER_MONTH["2026-01"] + GB_PER_MONTH["2026-07"],
        "ranking_months_kernel_hours": round(62 * sec / 3600, 1),
        "one_training_month_gb": GB_PER_MONTH["2025-07"],
        "one_training_month_kernel_hours": round(31 * sec / 3600, 1),
        "full_2025_gb": GB_PER_MONTH["2025-all"],
        "pilot_gb_transferred": round(pub / 1e9, 2),
        "pilot_gb_tar_member_payload": round(raw / 1e9, 2),
        "pilot_tar_overhead_mb": round((pub - raw) / 1e6, 1),
        "per_month_constants_agree": (
            "GB_PER_MONTH['2025-07'] / 31 = "
            f"{GB_PER_MONTH['2025-07'] / 31:.3f} GB per day against the pilot's measured "
            f"{gb:.3f}, so the per-month constants are already published asset bytes"),
    }
    (EXP / "analysis/extrapolate.json").write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
