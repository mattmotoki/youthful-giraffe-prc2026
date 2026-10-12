"""Derive exp-047-queue/analysis/audit-expect-<tag>.json for an exp-046 candidate.

Named for the JOB, not for the candidate it was first written for (section 47.54).  It was
`expectv24.py`, and a v25 written by copying it would have been section 47.43.4's defect at
FILE scale -- a whole tool describing the run it was cloned from.  One tool, one CANDIDATES
table, `--tag` picks the row.

The expectation file is what queue's audit is checked against, so every figure in it must
come from a file on disk -- never from a number typed twice.  This script takes v23's
finalised file as the TEMPLATE for the claims that do not change (the class, the template,
the pins: base set v24 changes what the head is FITTED on and nothing about which rows it
occupies), and refills every figure that does change from:

    * exp-046 analysis/serve-v24.json      -- the serve's own read and census
    * exp-046 analysis/threemonth-bag10.json -- the fold-side price and its registration
    * artifacts/exp-046/class/2026-{01,07}-v24.parquet -- the class vector, for the
      per-airport occupancy the serve report does not carry

It is validated by regenerating v23's file from v23's inputs and diffing against the
committed one: if the mapping is wrong there, it is wrong here.

    .venv/bin/python experiments/exp-046-surfacefit/code/expectv24.py --check   # reproduce v23
    .venv/bin/python experiments/exp-046-surfacefit/code/expectv24.py           # write v24
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
QUEUE = ROOT / "experiments/exp-047-queue"
FOLDS = ("july_loo", "december", "january")
FOLD_MONTH = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
EXPECTED_AIRPORTS = ("LFPG",)          # "the expected airports" of the v23 claims



# One row per candidate this tool has written. A new candidate ADDS a row; it never edits
# another candidate's (section 47.54).
TWO_HEAD_TAGS = ("v25",)
CANDIDATES = {
    "v24": {
        "prev": "v23", "candidate": "v10",
        "fold_price": "analysis/threemonth-bag10.json",
        "prediction_source": ("experiments/exp-046-surfacefit/RESULTS.md 47.33 "
                              "(commit 7c12959, before the fit)"),
        "what": ("candidate exp-046-surface-v10.parquet, exp-046 serve base set v24 = base "
                 "set v23 with ONE change: every fit on both the fold and the ranking side "
                 "carries the writable rows of 2025-11, 2025-10 AND 2025-03 (290,037) "
                 "instead of 2025-11's 99,709 alone."),
        "registered": {
            "registered_in": "exp-046 RESULTS.md 47.33, before the ten fits ran",
            "quantity": ("the K = 10 THREE-MONTH bag as an in-fold increment over v9's "
                         "K = 10 head"),
            "point_estimate_board_s": 0.468, "band_s": [0.36, 0.58],
            "every_fold_clear_expected": True, "fold_expected_to_fail": "january",
            "the_naive_figure_deliberately_not_registered": 0.612,
            "gate": ("in-fold, every fold clear, exp-046 rule (a) -- 47.33.1 verifies the "
                     "form was registered in 47.12 before its data, and 47.36 records that "
                     "the NEXT panel will not qualify")},
    },
    "v25": {
        "prev": "v24", "candidate": "v11",
        "fold_price": "analysis/seasonal-bag10.json",
        "prediction_source": ("experiments/exp-046-surfacefit/RESULTS.md 47.49.3 "
                              "(commit f435d25, before 2025-08 existed)"),
        "what": ("candidate exp-046-surface-v11.parquet, exp-046 serve base set v25 = base "
                 "set v24 with ONE change: the head is composed BY SERVED MONTH. 2026-01's "
                 "rows come from a head fitted on the folds plus 2025-11, 2025-10, 2025-03 "
                 "and 2025-02; 2026-07's from a head fitted on the folds plus 2025-11, "
                 "2025-10, 2025-03 and 2025-08. The served months are antipodal, so no one "
                 "month set serves both (D24, RESULTS 47.46)."),
        "registered": {
            "registered_in": ("exp-046 RESULTS.md 47.49.3, before 2025-08 existed and "
                              "before any Aug read"),
            "quantity": ("the K = 10 SEASONAL bag as an in-fold increment over v10's "
                         "K = 10 three-month head"),
            "point_estimate_board_s": 0.63115,
            "per_fold_point_estimates_s2_per_row": {
                "january": 224.90, "december": 66.01, "july_loo": 498.47},
            "every_fold_clear_expected": True,
            "fold_expected_to_decide_it": "december, at a scaled lower bound of +0.92",
            "gate": ("in-fold, every fold clear, exp-046 rule (a); read under D23's "
                     "discounted gate, credited = min(read, prediction)")},
    },
}


def _get(d: dict, path: str):
    cur = d
    for part in path.split("."):
        if part not in cur:
            raise SystemExit(f"serve report has no {path!r} (stopped at {part!r}) -- the "
                             f"schema moved and this mapping must be re-derived, not patched")
        cur = cur[part]
    return cur


def _how_to_run(srv: dict, tag: str, prev_tag: str) -> str:
    """The audit command for THIS candidate, every argument read from the serve report.

    Written rather than inherited: section 47.43.4.  The assertions below are positive (the
    line must name this run's own files) and one negative (it must not name the sibling the
    candidate replaces), so a line copied from the previous candidate cannot survive.
    """
    inc = "gate.increment_over_the_head"
    cand = _get(srv, "written_to")
    sibling = _get(srv, "sibling_check.file")
    bytes_file = _get(srv, "base.bytes_file")
    pins = _get(srv, "population.pin_file")
    held = _get(srv, "gate.fold.airports_held")
    line = (
        f"OMP_NUM_THREADS=2 ../../.venv/bin/python work/audit.py"
        f" --head-npz ../exp-046-surfacefit/work/{_get(srv, f'{inc}.head_vector')}"
        f" --arm-npz ../exp-046-surfacefit/work/{_get(srv, f'{inc}.arm_vector')}"
        f" --pins ../{Path(pins).relative_to('experiments')}"
        f" --class-vector ../../artifacts/exp-046/class/{{month}}-{tag}.parquet"
        f" --previous-class-vector ../../artifacts/exp-046/class/{{month}}-{prev_tag}.parquet"
        f" --candidate ../../{cand}"
        f" --previous ../../{bytes_file}"
        f" --baseline ../../{bytes_file}"
        f" --hold '{','.join(held)}'"
        f" --changed-expected-at {','.join(EXPECTED_AIRPORTS)}"
        f" --expect analysis/audit-expect-{tag}.json"
        f" --out analysis/audit-{tag}.json")
    for must in (Path(cand).name, _get(srv, f"{inc}.arm_vector"),
                 f"{{month}}-{tag}.parquet", f"audit-expect-{tag}.json", f"audit-{tag}.json"):
        if must not in line:
            raise SystemExit(f"how_to_run does not name {must!r} -- it does not describe "
                             f"this run and must not be written")
    if Path(sibling).name in line:
        raise SystemExit(f"how_to_run names the sibling {Path(sibling).name!r}, which is the "
                         f"candidate this one REPLACES -- that is the defect queue caught on "
                         f"audit-expect-v24.json (section 47.43.4)")
    return line


def _notes(srv: dict, tag: str, sibling_tag: str) -> str:
    """The three standing notes, written with THIS run's names (section 47.43.4)."""
    return (
        f"1. The class claims are hard, not carried over by habit: base set {tag} changes what "
        f"the head is FITTED on and nothing about which rows it serves, so occupied, written, "
        f"guard_held and the two class_vs_previous counts must come back identical to "
        f"{sibling_tag}'s. A difference there is a defect, not an update.\n"
        f"2. `increment.population.*.rows` are the scored populations at the pins and are "
        f"likewise unchanged; `rows_moved` counts rows where the ARM differs from the head, "
        f"and a differently fitted head moves a different number, so it is filled from "
        f"{Path(srv['written_to']).name.replace('.parquet', '')}'s own serve report.\n"
        f"3. {', '.join(EXPECTED_AIRPORTS)} is the expected airport for the same reason as on "
        f"{sibling_tag}: v18's head held it and this candidate does not.")

def _two_head_claim(srv: dict, tag: str) -> dict:
    """SECTION 47.54, queue's ask (1): the two-head claim, beside the extension.

    A seasonal candidate is TWO heads in one file.  The audit reads bytes, so it can see
    neither which rows each head wrote nor what each head was fitted on -- and the class
    vector's `head_by_exp_046` is queue's own stated limit: a DECLARATION, not a measurement.
    Everything below is therefore named so the auditor can rebuild each head itself: the ten
    member files and their shas per set, that set's months with their row counts, and the
    extra rows every fit actually carried.
    """
    ext = _get(srv, "extension")
    per_sm = ext["by_served_month"]
    info_sm = _get(srv, "fit.info.seasonal_by_served_month")
    out = {"0_what_this_is":
               f"candidate {tag} composes its ranking rows from TWO heads, one per served "
               f"month. `extension.composed_by` = {ext['composed_by']!r}. The audit cannot "
               f"see this in the bytes, so each set is named here in full",
           "0_fits_that_carried_it": ext["fits_that_carried_it"]}
    for sm in sorted(per_sm):
        decl = per_sm[sm]["declared"]
        pf = per_sm[sm]["per_fit"]
        if sm not in info_sm:
            raise SystemExit(f"the serve report declares an extension for {sm} but "
                             f"fit.info.seasonal_by_served_month has no {sm} -- the two "
                             f"halves of the claim disagree and neither may be written")
        files = info_sm[sm]["member_files"]
        shas = info_sm[sm]["member_sha256"]
        if sorted(files) != sorted(shas):
            raise SystemExit(f"{sm}: member_files and member_sha256 name different seeds")
        stems = {Path(v).name.rsplit("placebo", 1)[0] for v in files.values()}
        if len(stems) != 1:
            raise SystemExit(f"{sm}: its ten members do not share one cache stem: {stems}")
        if pf and len({(q["extra_rows"], q["total_fit_rows"]) for q in pf}) != 1:
            raise SystemExit(f"{sm}: extension.per_fit disagrees between fits")
        # SECTION 47.54.5, queue 7bw ii.  A mean of ten float64 arrays is SUMMATION-ORDER
        # dependent, so the order is part of the claim: the serve appends members in
        # `spec["seeds"]` order, and `member_files` above is emitted 1..K to match. A glob
        # order (1, 10, 2, ...) would reproduce the candidate only to the ULP class and
        # `max_abs_diff_s` would not be 0.
        seeds = info_sm[sm].get("seeds")
        if seeds is not None and list(seeds) != sorted(seeds):
            raise SystemExit(f"{sm}: the serve stacked its members in {list(seeds)} order, "
                             f"which is not ascending. The expectation file would tell the "
                             f"audit to sum in an order the candidate was not built in")
        out[sm] = {
            "months": decl["months"],
            "month_rows": decl.get("month_rows"),
            "shas_checked_before_the_fit": decl.get("shas"),
            "extension_rows_built": per_sm[sm]["rows_built"],
            "ranking_rows_this_head_wrote": per_sm[sm]["rows_this_head_writes"],
            "member_files": [files[k] for k in sorted(files, key=int)],
            "member_sha256": {k: shas[k] for k in sorted(shas, key=int)},
            "per_fit_extra_rows": (None if not pf else
                                   {"fits": len(pf), "extra_rows": pf[0]["extra_rows"],
                                    "fold_rows": pf[0]["fold_rows"],
                                    "total_fit_rows": pf[0]["total_fit_rows"],
                                    "multiple": pf[0]["multiple"]}),
            "cache_stem": stems.pop(),
            "member_order": ("NUMERIC, seed 1 first through seed 10 -- the order the serve "
                             "stacked them in. A mean of ten float64 arrays depends on the "
                             "summation order, so glob order (1, 10, 2, ...) reproduces the "
                             "candidate only to the ULP class"),
            "what": (f"the head that wrote candidate {tag}'s {sm} rows: fitted on the folds "
                     f"plus {', '.join(decl['months'])}, bagged over the ten members above")}
    stems = {out[sm]["cache_stem"] for sm in per_sm}
    if len(stems) != len(per_sm):
        raise SystemExit(f"the served months share a cache stem {stems} -- the key is "
                         f"seed+leaves (RESULTS 45.2) and a shared stem means one head was "
                         f"served twice. The claim would be false and is not written")
    return out


def _negative_control(srv: dict, tag: str) -> dict:
    """SECTION 47.54, queue's ask (2): the control that fails if the heads are the same.

    "The heads differ" is a boolean, and an auditor cannot reproduce a boolean.  This is the
    same claim as a DISTANCE, per served month, on the rows that head actually wrote.
    """
    info_sm = _get(srv, "fit.info.seasonal_by_served_month")
    sep = {}
    for sm in sorted(info_sm):
        s_here = info_sm[sm].get("separation_on_its_own_rows")
        if not s_here:
            raise SystemExit(f"{sm}: the serve report carries no separation measurement, so "
                             f"queue's negative control cannot be written as a number")
        for other, v in s_here.items():
            if v["rms_s"] <= 0.0 or v["rows_that_differ"] == 0:
                raise SystemExit(
                    f"{sm} vs {other}: the two heads are IDENTICAL on {sm}'s rows. The "
                    f"candidate is not seasonal and no expectation file is written")
            sep[f"{sm}_vs_{other}"] = {
                "rows": v["rows"], "rms_s": v["rms_s"], "max_abs_s": v["max_abs_s"],
                "rows_that_differ": v["rows_that_differ"]}
    return {
        "1_rebuild_each_set":
            "mean the ten member vectors named per served month above, IN THE ORDER THEY "
            "ARE LISTED -- numeric, seed 1 through seed 10, which is the order the serve "
            "stacked them in. A mean of ten float64 arrays is summation-order dependent, so "
            "glob order (1, 10, 2, ...) reproduces the candidate only to the ULP class. In "
            "numeric order a bag rebuilt from STORED arrays is exact (exp-046 RESULTS "
            "47.11's audit table), so this step carries no tolerance",
        "1a_what_the_numbers_must_then_be":
            "max_abs_diff_s EXACTLY 0.0 on the rows the class marks written, and "
            "rows_differing_above_tolerance (1e-9) = 0. The tolerance is what carries the "
            "claim; the exact zero is what says the order was right",
        "2_compose_by_the_MOVEMENT_TABLE_not_by_the_declaration":
            "take each ranking row's served month from data/raw/ranking.parquet, not from "
            "the class vector's `head_by_exp_046`. That column is a DECLARATION -- queue's "
            "own limit -- and this control is what turns it into a measurement",
        "3_the_positive_half":
            f"price base + the composed mean must reproduce candidate {tag} EXACTLY on every "
            f"row the class marks written_by_exp_046; the candidate is float64 and the "
            f"rebuild is a mean of stored float64 arrays, so exact means exact. Rows the "
            f"tail guard returned to the base, and rows outside the class, must equal the "
            f"BYTES BASE byte for byte -- the `--baseline`/`--previous` file, NOT the price "
            f"base. The two are different files (v25: bytes youthful-giraffe_v18.parquet, "
            f"price exp-035-k2climb-v11member.parquet) and they agree only where the k2 "
            f"lane left the row alone. ROUND 13 measured it on v11: all 148,585 rows "
            f"outside the class equal the bytes base and only 78,951 of them equal the "
            f"price base, so the earlier wording was false on 69,634 rows. The 1,573 "
            f"guard-held rows equal BOTH, which is why the error hid: the guard half of "
            f"the sentence was true either way",
        "4_the_negative_half":
            "price base + the mean of the OTHER set, restricted to a served month's own "
            "rows, must NOT reproduce the candidate there. The distance is measured, not "
            "asserted, and it is:",
        "4_separation": sep,
        "5_and_then_the_declaration_is_measured":
            "with 3 and 4 both holding, `head_by_exp_046` may be compared to the served "
            "month derived in step 2. Agreement on every row is what promotes the column "
            "from a declaration to a measurement; disagreement on any row is a defect in "
            "the candidate, not in the audit"}


def _extension_note(srv: dict, tag: str, dry: dict | None) -> dict:
    """What the audit cannot see: the extension the head was FITTED on, from the record.

    A serve that read its ranking members from cache reports `per_fit` EMPTY and
    `fits_that_carried_it` 0 -- that is the report telling the truth (section 47.22.2), and
    the fits it is describing are in the DRY RUN's report.  This falls back to that report and
    says so, rather than treating an empty list as an absent extension.
    """
    if _get(srv, "extension").get("composed_by") == "served month":
        # SECTION 47.54: a seasonal candidate has one extension PER SERVED MONTH and no
        # single `per_fit`. The claim is not weaker, it is bigger, and it is written by
        # `_two_head_claim`; this key says where it went rather than half-answering here.
        return {"1_this_candidate_has_TWO_extensions":
                    "one per served month -- see `the_two_head_claim_audit_py_cannot_see`",
                "2_why_the_audit_cannot_see_it":
                    "the audit reads BYTES. Neither what a head was fitted on nor which "
                    "rows it wrote leaves a trace in the parquet, so both are checked "
                    "against the serve report and against the negative control"}
    per = _get(srv, "extension.per_fit")
    members = sorted(Path(v).name for v in _get(srv, "fit.info.member_files").values())
    stem = {Path(m).name.rsplit("placebo", 1)[0] for m in members}
    if len(stem) != 1:
        raise SystemExit(f"the ranking members do not share one cache stem: {sorted(stem)}")
    fitted_by = f"serve-{tag}.json"
    if not per:
        if _get(srv, "extension.fits_that_carried_it") != 0 or dry is None:
            raise SystemExit("extension.per_fit is empty and no dry-run report was given -- "
                             "the extension claim cannot be written from the record")
        per = _get(dry, "extension.per_fit")
        fitted_by = f"serve-{tag}-dryrun.json (the real serve refitted nothing and read the "
        fitted_by += "ten caches the dry run wrote; its own per_fit is EMPTY, which is the "
        fitted_by += "report telling the truth)"
    if len({(q["extra_rows"], q["total_fit_rows"]) for q in per}) != 1:
        raise SystemExit("extension.per_fit disagrees between fits -- the extension claim "
                         "cannot be written from the record")
    decl = _get(srv, "extension.declared")
    # a single-month extension declares `month`, a stack declares `months`; both are the
    # record and neither is inferred
    months = decl["months"] if "months" in decl else [decl["month"]]
    return {
        "1_ranking_members_are_extended":
            f"serve-{tag}.json fit.info.member_files must be {len(members)} "
            f"work/{stem.pop()}placebo{{seed}}-255.npz. The cache key is seed+leaves and "
            f"cannot tell one ARM from another (exp-046 RESULTS 45.2), so the distinct stem "
            f"is what keeps this candidate's members apart from the previous one's",
        "2_every_ranking_fit_grew":
            f"the fits are reported by {fitted_by}: per_fit has {len(per)} entries, "
            f"each extra_rows = "
            f"{per[0]['extra_rows']:,}, fold_rows {per[0]['fold_rows']:,} -> total_fit_rows "
            f"{per[0]['total_fit_rows']:,} ({per[0]['multiple']}x); "
            f"fits_that_carried_it = {_get(srv, 'extension.fits_that_carried_it')}",
        "3_which_months":
            f"the extension is {', '.join(months)} = "
            f"{decl['rows']:,} writable rows, shas in extension.declared, each "
            f"checked before the fit",
        "4_why_the_audit_cannot_see_it":
            "the audit reads BYTES. What the head was fitted on leaves no trace in the "
            "parquet, so these three claims are checked against the serve report, not "
            "against the candidate"}


def build(serve_path: Path, class_files: dict, tag: str, prev_tag: str,
          template: Path) -> dict:
    srv = json.loads(serve_path.read_text())
    out = json.loads(template.read_text())
    inc = "gate.increment_over_the_head"

    e = out["expect"]
    for f in FOLDS:
        e[f"pins.by_fold.{f}.their_airport_days"] = _get(
            srv, f"population.fold_months_are_pinned_to.{FOLD_MONTH[f]}.airport_days")
        e[f"increment.population.{f}.rows"] = _get(srv, f"{inc}.by_fold.{f}.all_row.rows")
        e[f"increment.population.{f}.rows_moved"] = _get(
            srv, f"{inc}.by_fold.{f}.all_row.rows_moved")
        e[f"increment.per_fold.{f}.mse_gain_s2"] = {
            "value": _get(srv, f"{inc}.by_fold.{f}.all_row.mse_gain_s2"), "tol": 1e-09}
        e[f"increment.per_fold.{f}.ci95_envelope_over_the_seeds"] = {
            "inside": _get(srv, f"{inc}.by_fold.{f}.all_row.ci95"), "tol": 3.0}
    e["increment.board.all_row_gain_s"] = {
        "value": _get(srv, f"{inc}.board_all_row_gain_s"), "tol": 1e-09}
    e["increment.board.clean_gain_s"] = {
        "value": _get(srv, f"{inc}.board_clean_gain_s"), "tol": 1e-09}
    e["increment.board.all_row_ci95_envelope_over_the_seeds"] = {
        "inside": _get(srv, f"{inc}.board_all_row_interval_ci95"), "tol": 0.0}

    e["bytes.candidate_sha256"] = _get(srv, "sha256")
    e["bytes.moved.rows_changed"] = _get(srv, "census.moved")
    e["bytes.template.rows"] = _get(srv, "census.handover_to_exp_047.rows")
    e["bytes.class.noac_rows"] = _get(srv, "census.noac_rows")
    e["bytes.class.occupied"] = _get(srv, "census.handover_to_exp_047.served_rows")
    e["bytes.class.written"] = _get(srv, "census.handover_to_exp_047.served_rows")
    e["bytes.class.guard_held"] = _get(srv, "gate.tail_guard.rows_returned_to_the_base")

    # the per-airport split the serve report does not carry: read it from the class vector
    moved_at = sum(_get(srv, f"census.by_airport.{a}.moved") for a in EXPECTED_AIRPORTS)
    # the class vector is keyed by movement id and carries no airport, so the airport comes
    # from the movement table -- the same join the serve itself uses
    # 2026-01 and 2026-07 are RANKING months, not training months, so the airport comes
    # from data/raw/ranking.parquet -- the table the serve itself writes against
    rank = pd.read_parquet(ROOT / "data/raw/ranking.parquet",
                           columns=["MVT_ID_mvt", "ADEP_mvt"])
    rank["MVT_ID_mvt"] = rank["MVT_ID_mvt"].astype("float64")
    adep = rank.drop_duplicates("MVT_ID_mvt").set_index("MVT_ID_mvt").ADEP_mvt
    occ_at = 0
    held_at = 0
    for month, path in class_files.items():
        c = pd.read_parquet(path)
        ap = adep.reindex(c.MVT_ID_mvt.to_numpy("float64"))
        if ap.isna().any():
            raise SystemExit(f"{path}: {int(ap.isna().sum())} class rows have no airport in "
                             f"ranking.parquet -- the per-airport claims cannot be derived")
        at = c[ap.isin(EXPECTED_AIRPORTS).to_numpy()]
        occ_at += int(at["occupied_by_exp_046"].sum())
        held_at += int(at["guard_held_by_exp_046"].sum())
    e["bytes.class.writable_at_the_expected_airports"] = moved_at
    e["bytes.moved.changed_rows_at_the_expected_airports"] = moved_at
    e["bytes.class.occupied_at_the_expected_airports"] = occ_at
    e["bytes.class.guard_held_at_the_expected_airports"] = occ_at - moved_at
    e["bytes.moved.changed_rows_outside_the_expected_airports"] = (
        _get(srv, "census.moved") - moved_at)
    if held_at and held_at != occ_at - moved_at:
        raise SystemExit(f"the class vector says {held_at} guard-held rows at "
                         f"{EXPECTED_AIRPORTS} but occupied-minus-writable says "
                         f"{occ_at - moved_at}; the two disagree and one is wrong")

    # SECTION 47.43.4.  Every field that DESCRIBES THIS RUN is written from the record.
    # v24's first emission inherited v23's how_to_run verbatim from the template, so the file
    # told queue to audit v9 against v23's expectations -- a section 45 silent default that
    # survived because the template's prose is copied while only its numbers are refilled.
    out["how_to_run"] = _how_to_run(srv, tag, prev_tag)
    dry_path = serve_path.with_name(serve_path.stem + "-dryrun.json")
    out["notes_on_the_expectation"] = _notes(
        srv, tag, Path(_get(srv, "sibling_check.file")).stem.rsplit("-", 1)[-1])
    out["the_extension_audit_py_cannot_see"] = _extension_note(
        srv, tag, json.loads(dry_path.read_text()) if dry_path.exists() else None)
    out["sources"] = dict(out["sources"])
    out["sources"]["serve"] = str(serve_path.relative_to(ROOT))
    out["finalised_from"] = {
        "serve_report": str(serve_path.relative_to(ROOT)),
        "candidate_sha256": _get(srv, "sha256"),
        "board_all_row_gain_s": _get(srv, f"{inc}.board_all_row_gain_s"),
        "every_fold_interval_clear": all(
            _get(srv, f"{inc}.by_fold.{f}.all_row.ci95")[0] > 0 for f in FOLDS)}
    return out


def main() -> None:
    """`--check` reproduces v23's committed file; `--tag vNN` writes that candidate's."""
    check = "--check" in sys.argv
    if check:
        got = build(EXP / "analysis/serve-v23.json",
                    {"2026-01": ROOT / "artifacts/exp-046/class/2026-01-v20.parquet",
                     "2026-07": ROOT / "artifacts/exp-046/class/2026-07-v20.parquet"},
                    "v23", "v22", QUEUE / "analysis/audit-expect-v23.json")
        want = json.loads((QUEUE / "analysis/audit-expect-v23.json").read_text())
        bad = {k: (want["expect"][k], got["expect"][k])
               for k in want["expect"] if want["expect"][k] != got["expect"][k]}
        if bad:
            print(f"MAPPING IS WRONG on {len(bad)} of {len(want['expect'])} expectations:")
            for k, (w, g) in list(bad.items())[:12]:
                print(f"  {k}\n      committed {w!r}\n      derived   {g!r}")
            raise SystemExit(1)
        if got["how_to_run"] != want["how_to_run"]:
            print("HOW_TO_RUN IS WRONG:")
            print(f"  committed {want['how_to_run']}")
            print(f"  derived   {got['how_to_run']}")
            raise SystemExit(1)
        print(f"mapping reproduces all {len(want['expect'])} of v23's expectations from "
              f"serve-v23.json and the class vector, and its hand-written how_to_run line "
              f"character for character")
        return

    tag = "v25" if "--tag" not in sys.argv else sys.argv[sys.argv.index("--tag") + 1]
    if tag not in CANDIDATES:
        raise SystemExit(f"{tag} is not a known candidate; known: {sorted(CANDIDATES)}")
    spec = CANDIDATES[tag]
    srv = EXP / f"analysis/serve-{tag}.json"
    if not srv.exists():
        raise SystemExit(f"{srv} does not exist yet -- the {spec['candidate']} serve has "
                         f"not run, and no expectation file may be written before it")
    out = build(srv, {m: ROOT / f"artifacts/exp-046/class/{m}-{tag}.parquet"
                      for m in ("2026-01", "2026-07")},
                tag, spec["prev"], QUEUE / f"analysis/audit-expect-{spec['prev']}.json")
    srv_j = json.loads(srv.read_text())
    bag_path = EXP / spec["fold_price"]
    bag = json.loads(bag_path.read_text())
    out["sources"]["fold_price"] = str(bag_path.relative_to(ROOT))
    out["sources"]["prediction"] = spec["prediction_source"]
    out["what"] = spec["what"]
    out["status"] = (f"FINAL. Filled from analysis/serve-{tag}.json and the {tag} class "
                     f"vector after the serve landed. No PENDING, no null.")
    out["the_registered_prediction"] = dict(spec["registered"], **{
        "measured": bag["verdict"]["board_gain_s"],
        "inside_the_band": bag["verdict"].get("inside_the_registered_band")})

    # SECTION 47.54, queue's two asks. Written only for a candidate whose own report says it
    # is composed by served month, so a single-head candidate cannot acquire the claim by
    # inheriting this file from a seasonal one.
    if srv_j.get("extension", {}).get("composed_by") == "served month":
        out["the_two_head_claim_audit_py_cannot_see"] = _two_head_claim(srv_j, tag)
        out["the_negative_control"] = _negative_control(srv_j, tag)
    elif tag in TWO_HEAD_TAGS:
        raise SystemExit(f"{tag} is registered as a seasonal candidate but its serve report "
                         f"is not composed by served month -- one of the two is wrong")

    nulls = [k for k, v in out["expect"].items() if v is None]
    pend = [k for k, v in out["expect"].items() if v == "PENDING"]
    if nulls or pend:
        raise SystemExit(f"not final: {len(nulls)} null, {len(pend)} PENDING")
    out["pending"], out["pending_because"] = [], (
        f"nothing is pending: every figure is filled from analysis/serve-{tag}.json, the "
        f"{tag} class vector and {spec['fold_price']} by code/expectfile.py")
    # `--to PATH` writes somewhere else. It exists so a REGENERATION can be diffed against
    # the committed file without touching another lane's live directory (section 47.54).
    dst = (Path(sys.argv[sys.argv.index("--to") + 1]) if "--to" in sys.argv
           else QUEUE / f"analysis/audit-expect-{tag}.json")
    dst.write_text(json.dumps(out, indent=1, sort_keys=True))
    shown = dst.relative_to(ROOT) if dst.is_relative_to(ROOT) else dst
    print(f"written {shown}: {len(out['expect'])} expectations, "
          f"0 pending, 0 null"
          + (", two-head claim and negative control carried"
             if "the_negative_control" in out else ""))


if __name__ == "__main__":
    main()
