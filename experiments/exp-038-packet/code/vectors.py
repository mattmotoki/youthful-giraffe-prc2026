"""Every fold-side predictor vector this packet rebases, and the gate on each.

**The reading taken.**  `assignments/packet.md` says "every served or staged
vector since stack-5" and then names nine families.  The colon list is read as
naming the families, not as closing the set: `exp-030-window2`'s three staged
window arms are also vectors staged since stack-5, they cost nothing to price
because their rotations are committed, and leaving them out would make the
inventory's own title false.  They are marked `staged-extra` in the table so a
reader can drop them.  Readings rejected: (a) treating the colon list as
exhaustive -- it would omit vectors the sentence before it includes; (b) pricing
member-level OOF columns (`bulk64`, `k2s1`, `k3-tail-*`) as candidates -- a
member is not a predictor and Q01's `d` is defined on the served vector.

**Nothing is refitted here.**  Every vector is either read back from a committed
rotation or built by `np.where` over two of them, and each is gated against the
number its own experiment published before it enters the table.  Where a vector
cannot be reproduced to its published number it is dropped with the reason
printed, never rebuilt (`assignments/packet.md`).

The baseline is **v10**, `submissions/uploaded/youthful-giraffe_v10.parquet`,
SHA-256 `ad962763…`, which is exp-036 arm 2: the refit-bulk rotation served at
the eight airports of `experiments/exp-036-mask/analysis/mask.json` and v7
elsewhere.  Its fold-side form is rebuilt here by that same arithmetic and gated
on exp-036's published clean rotation (284.1572 / 204.9218 / 218.1506).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
sys.path.insert(0, str(ROOT / "experiments/exp-030-window2/code"))
import score as Sc  # noqa: E402
sys.path.insert(0, str(ROOT))
import prc.data as D  # noqa: E402

FOLDS = Sc.FOLDS
SHARES = Sc.P.SHARES
GATE = 1e-3
VEC_TOL = 1e-9
EHAM = "EHAM"

V10_FILE = "submissions/uploaded/youthful-giraffe_v10.parquet"
V10_SHA = "ad962763236e9b40ace45a5dac12945568bbce441ed0547ff8e814d248c72c78"
MASK_JSON = "experiments/exp-036-mask/analysis/mask.json"

#: `(oof path template, column, published clean per fold, provenance)`.  Every
#: `published` here is copied from the named experiment's own analysis record.
DIRECT = {
    "K2 (v9)": ("experiments/exp-027-step/oof/exp-027-k2-k2-hold-{fold}.parquet", "y_pred",
                {"july_loo": 284.3195, "december": 205.2156, "january": 218.2582},
                "exp-027 analysis/candidate-k2.json arm k2-hold-eham-out; board 263.0827"),
    "K2 no-EHAM-mask": ("experiments/exp-027-step/oof/exp-027-k2-k2-hold-{fold}.parquet",
                        "y_pred_nomask",
                        {"july_loo": 284.3938, "december": 205.2990, "january": 218.4232},
                        "exp-027 arm k2-hold (unmasked)"),
    "refit-bulk": ("experiments/exp-027-step/oof/exp-027-k2-refit-bulk-{fold}.parquet", "y_pred",
                   {"july_loo": 284.1774, "december": 204.9052, "january": 218.2115},
                   "exp-027 analysis/refit-bulk.json arm k2-refit-bulk-eham-out; never uploaded"),
    "refit-bulk no-EHAM-mask": ("experiments/exp-027-step/oof/exp-027-k2-refit-bulk-{fold}.parquet",
                                "y_pred_nomask", None, "exp-027 refit-bulk (unmasked)"),
    "k3-hold": ("experiments/exp-032-k3/oof/exp-032-k3-k3-hold-{fold}.parquet", "y_pred",
                {"july_loo": 284.4265, "december": 205.3721, "january": 218.3111},
                "exp-032 analysis/candidate-k3.json arm k3-hold-eham-out"),
    "k3g-hold": ("experiments/exp-032-k3/oof/exp-032-k3-k3g-hold-{fold}.parquet", "y_pred",
                 {"july_loo": 284.4235, "december": 205.3879, "january": 218.3125},
                 "exp-032 arm k3g-hold-eham-out (gated K3 member)"),
    "k3-hold no-EHAM-mask": ("experiments/exp-032-k3/oof/exp-032-k3-k3-hold-{fold}.parquet",
                             "y_pred_nomask", None, "exp-032 k3-hold (unmasked)"),
    "k3g-hold no-EHAM-mask": ("experiments/exp-032-k3/oof/exp-032-k3-k3g-hold-{fold}.parquet",
                              "y_pred_nomask", None, "exp-032 k3g-hold (unmasked)"),
    "bulk-add-hold": ("experiments/exp-034-bulk/oof/exp-034-bulk-bulk-add-hold-{fold}.parquet",
                      "y_pred",
                      {"july_loo": 284.3591, "december": 205.1995, "january": 218.2214},
                      "exp-034 analysis/candidate-bulk.json arm bulk-add-hold-eham-out"),
    "bulk-add-bulkonly-hold": (
        "experiments/exp-034-bulk/oof/exp-034-bulk-bulk-add-bulkonly-hold-{fold}.parquet",
        "y_pred", {"july_loo": 284.3493, "december": 205.2080, "january": 218.2338},
        "exp-034 arm bulk-add-bulkonly-hold-eham-out"),
    "bulk-add-nmbulk-out": (
        "experiments/exp-034-bulk/oof/exp-034-bulk-bulk-add-nmbulk-out-{fold}.parquet", "y_pred",
        {"july_loo": 284.3170, "december": 205.2509, "january": 218.2280},
        "exp-034 exploratory arm bulk-add-nmbulk-out-eham-out"),
    "bulkonly-nmbulk-out": (
        "experiments/exp-034-bulk/oof/exp-034-bulk-bulkonly-nmbulk-out-{fold}.parquet", "y_pred",
        {"july_loo": 284.3308, "december": 205.2091, "january": 218.2434},
        "exp-034 exploratory arm bulkonly-nmbulk-out-eham-out"),
    "k2-w300": ("experiments/exp-036-mask/oof/exp-036-k2-w300-{fold}.parquet", "y_pred",
                {"july_loo": 284.2158, "december": 205.2253, "january": 218.2452},
                "exp-036 analysis/arm-k2-w300.json (600 s + 300 s widths, EHAM mask)"),
    "k2-w300 no-EHAM-mask": ("experiments/exp-036-mask/oof/exp-036-k2-w300-{fold}.parquet",
                             "y_pred_nomask", None, "exp-036 arm 3 unmasked"),
    "w600arrk (staged-extra)": (
        "experiments/exp-030-window2/oof/exp-028-stack5w-w600arrk-{fold}.parquet", "y_pred",
        {"july_loo": 284.7857, "december": 205.7016, "january": 218.7775},
        "exp-030 analysis/arm-w600arrk.json (unmasked)"),
    "w600w300 (staged-extra)": (
        "experiments/exp-030-window2/oof/exp-028-stack5w-w600w300-{fold}.parquet", "y_pred",
        {"july_loo": 284.6135, "december": 205.6432, "january": 218.7987},
        "exp-030 analysis/arm-w600w300.json (unmasked)"),
    "w600w300w1800 (staged-extra)": (
        "experiments/exp-030-window2/oof/exp-028-stack5w-w600w300w1800-{fold}.parquet", "y_pred",
        {"july_loo": 284.5221, "december": 205.8630, "january": 218.6994},
        "exp-030 analysis/arm-w600w300w1800.json (unmasked)"),
}

#: exp-037's chain heads.  Each is v10 with the fold's `noac` rows replaced by
#: the arm's own prediction; nothing else in the vector moves.  `work/
#: repair_eval.parquet` carries the eight repair arms and `work/fold_eval`'s
#: `arm` column the raw chain correction; the two blend heads are rebuilt from
#: `code/blend.py`'s own outputs where they exist.
E37_REPAIR = "experiments/exp-037-tail/work/repair_eval.parquet"
#: `code/repair.py`: two self-fitted heads (`chain`, `rot`) x four carriers
#: (`full`, `clf`, `reg`, `clip`) x raw and out-of-fold-shrunk.  Each column
#: already carries exp-037's own `MIN_TAXI` floor, so it is the arm's served
#: `noac` value and not a correction.
E37_ARMS = tuple(f"{h}_{k}_{c}" for h in ("chain", "rot")
                 for k in ("raw", "shr") for c in ("full", "clf", "reg", "clip"))


def sha_vec(v: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(v, dtype="float64").tobytes()).hexdigest()


def _order(d: pd.DataFrame, ref: pd.DataFrame, what: str, fold: str) -> None:
    if len(d) != len(ref) or not np.array_equal(d["MVT_ID_mvt"].to_numpy("float64"),
                                                ref["MVT_ID_mvt"].to_numpy("float64")):
        raise SystemExit(f"{fold}: {what} is not the reference's one-to-one movement-id join")


def clean_of(vec: dict, ref: dict) -> dict:
    return {f: D.rmse_clean(ref[f]["y_true"], vec[f], ref[f]["MVT_ID_mvt"], f) for f in FOLDS}


def eham_mask(preds: dict, ref: dict) -> dict:
    return {f: np.where(ref[f]["airport"].astype(str).to_numpy() == EHAM,
                        ref[f]["y_pred_stack5"].to_numpy("float64"), preds[f]) for f in FOLDS}


def load(verbose: bool = True) -> tuple[dict, dict, dict, dict]:
    """`(reference panel, {name: {fold: vector}}, {name: provenance}, files)`."""
    ref = Sc.reference()                    # gates stack-5 and v6 on exp-028's rotation
    keep = json.loads((ROOT / MASK_JSON).read_text())["keep"]
    vec, prov, files, dropped = {}, {}, {}, {}

    def gate(name: str, v: dict, pub, src: str, extra: dict | None = None) -> bool:
        got = clean_of(v, ref)
        if pub is not None:
            off = {f: round(got[f] - pub[f], 4) for f in FOLDS}
            if max(abs(x) for x in off.values()) > GATE:
                dropped[name] = f"does not reproduce its published rotation: {off}"
                if verbose:
                    print(f"[drop] {name}: {dropped[name]}")
                return False
        vec[name] = v
        prov[name] = {"source": src, "clean": {f: round(got[f], 4) for f in FOLDS},
                      "clean_board": round(Sc.P.board(got), 4),
                      "published": pub, "gated": pub is not None,
                      "sha256_by_fold": {f: sha_vec(v[f]) for f in FOLDS}}
        if extra:
            prov[name].update(extra)
        return True

    # -- the two vectors `score.reference` has already gated, and v7 --------
    gate("stack-5", {f: ref[f]["y_pred_stack5"].to_numpy("float64") for f in FOLDS},
         Sc.PUBLISHED["stack-5"], "exp-026 stack5.PUBLISHED, via exp-030 score.reference")
    gate("v6", {f: ref[f]["y_pred_v6"].to_numpy("float64") for f in FOLDS},
         Sc.PUBLISHED["v6"], "exp-028 600 s window stacker, via exp-030 score.reference")
    v7 = eham_mask(vec["v6"], ref)
    gate("v7", v7, {"july_loo": 284.6251, "december": 205.5677, "january": 218.5412},
         "exp-030 arm 3 (v6 with EHAM from stack-5), rebuilt by np.where; board 263.4623")

    # -- the committed rotations -------------------------------------------
    for name, (path, col, pub, src) in DIRECT.items():
        try:
            v = {}
            for f in FOLDS:
                d = pd.read_parquet(ROOT / path.format(fold=f))
                _order(d, ref[f], name, f)
                v[f] = d[col].to_numpy("float64")
        except (FileNotFoundError, KeyError) as e:
            dropped[name] = f"absent or lacks its column: {e}"
            if verbose:
                print(f"[drop] {name}: {dropped[name]}")
            continue
        files[name] = {f: str(Path(path.format(fold=f))) for f in FOLDS}
        # An unmasked variant has no published clean triple of its own.  Its gate
        # is the identity that applying v7's EHAM mask to it reproduces its
        # published masked sibling ROW BY ROW, which is a statement about the
        # same file's two columns and therefore checkable.
        if pub is None and name.endswith("no-EHAM-mask"):
            sib = name[: -len(" no-EHAM-mask")]
            if sib not in vec:
                dropped[name] = f"its masked sibling {sib!r} did not gate"
                if verbose:
                    print(f"[drop] {name}: {dropped[name]}")
                continue
            m = eham_mask(v, ref)
            off = {f: float(np.abs(m[f] - vec[sib][f]).max()) for f in FOLDS}
            if max(off.values()) > VEC_TOL:
                dropped[name] = (f"the EHAM mask on it does not reproduce {sib}: "
                                 f"max abs row difference {max(off.values()):g}")
                if verbose:
                    print(f"[drop] {name}: {dropped[name]}")
                continue
            gate(name, v, None, src, {"gated": True,
                 "gated_on": f"EHAM mask on it reproduces {sib} row by row",
                 "max_abs_row_difference": max(off.values())})
            continue
        gate(name, v, pub, src)

    # -- the derived masks -------------------------------------------------
    def apt(f):
        return ref[f]["airport"].astype(str).to_numpy()

    for name, src_name, pub, note in (
        ("k2-mask", "K2 (v9)", {"july_loo": 284.3193, "december": 205.2376, "january": 218.2457},
         "exp-036 arm 1: K2 served at the eight kept airports, v7 elsewhere"),
        ("v10 (exp-036 refitbulk-mask)", "refit-bulk",
         {"july_loo": 284.1572, "december": 204.9218, "january": 218.1506},
         "exp-036 arm 2 = THE BASELINE: refit-bulk at the eight kept airports, v7 elsewhere"),
    ):
        if src_name not in vec:
            continue
        v = {f: np.where(np.isin(apt(f), keep), vec[src_name][f], v7[f]) for f in FOLDS}
        gate(name, v, pub, note, {"mask_keep": list(keep)})

    # -- exp-037's chain heads, on v10's noac rows -------------------------
    base = vec.get("v10 (exp-036 refitbulk-mask)")
    if base is not None and (ROOT / E37_REPAIR).exists():
        rp = pd.read_parquet(ROOT / E37_REPAIR)
        ra = pd.read_csv(ROOT / "experiments/exp-037-tail/analysis/repair_arms.csv")
        for arm in E37_ARMS:
            if arm not in rp.columns:
                continue
            v, moved = {}, {}
            ok = True
            for f in FOLDS:
                sub = rp[rp["fold"] == f]
                ids = ref[f]["MVT_ID_mvt"].to_numpy("float64")
                pos = pd.Index(ids).get_indexer(sub["MVT_ID_mvt"].to_numpy("float64"))
                if (pos < 0).any():
                    ok = False
                    break
                w = base[f].copy()
                # the arm must agree with v10 on the rows it does not claim
                if float(np.abs(sub["shipped"].to_numpy("float64") - w[pos]).max()) > VEC_TOL:
                    ok = False
                    break
                w[pos] = sub[arm].to_numpy("float64")
                moved[f] = int((w[pos] != base[f][pos]).sum())
                v[f] = w
            if not ok:
                dropped[f"exp-037 {arm}"] = "does not join v10 one-to-one on the noac rows"
                if verbose:
                    print(f"[drop] exp-037 {arm}: {dropped[f'exp-037 {arm}']}")
                continue
            # exp-037's own gate: the arm's RMSE on the fold's `noac` rows, from
            # `analysis/repair_arms.csv`.  A published clean-board number does not
            # exist for these arms (no candidate was written), so the reproduction
            # is checked where exp-037 measured them.
            head, kind, carrier = arm.split("_")
            off = {}
            for f in FOLDS:
                m = ra[(ra["head"] == head) & (ra["arm"] == carrier)
                       & (ra["kind"] == kind) & (ra["fold"] == f)]
                nz = ref[f]["regime"].to_numpy() == "noac"
                got = float(np.sqrt(np.mean(
                    (ref[f]["y_true"].to_numpy("float64")[nz] - v[f][nz]) ** 2)))
                off[f] = round(got - float(m["rmse_arm"].iloc[0]), 4)
            if max(abs(x) for x in off.values()) > GATE:
                dropped[f"exp-037 {arm}"] = f"noac RMSE does not reproduce exp-037: {off}"
                if verbose:
                    print(f"[drop] exp-037 {arm}: {dropped[f'exp-037 {arm}']}")
                continue
            gate(f"exp-037 {arm}", v, None,
                 "exp-037 work/repair_eval.parquet: v10 with the noac rows served by this "
                 "chain arm; exp-037 RESULTS section 5 (no candidate parquet was written)",
                 {"noac_rows_moved": moved, "gated": True,
                  "gated_on": "exp-037 repair_arms.csv rmse_arm (noac rows, per fold)",
                  "noac_rmse_offset": off})

    # -- what could not be priced -----------------------------------------
    dropped["exp-035 seed arm"] = (
        "no head-level fold vector exists: oof/ carries the MEMBER columns k2s1 (3 folds), "
        "k2s2 (july_loo only) and k2w (july_loo only), the seed-33 december/january member "
        "fits were still on Kaggle at 2026-09-08 (NEXT.md), and rotate.py --arm seeds has "
        "not run. Rebuilding it is a member fit, which assignments/packet.md forbids")

    files["v10_served"] = {"path": V10_FILE, "sha256": D.sha256(ROOT / V10_FILE),
                           "expected": V10_SHA}
    if files["v10_served"]["sha256"] != V10_SHA:
        raise SystemExit(f"{V10_FILE} has changed: {files['v10_served']['sha256'][:8]}")
    files["mask"] = {"path": MASK_JSON, "sha256": D.sha256(ROOT / MASK_JSON), "keep": list(keep)}
    files["dropped"] = dropped
    return ref, vec, prov, files


if __name__ == "__main__":
    ref, vec, prov, files = load()
    print(f"\n{len(vec)} vectors gated; {len(files['dropped'])} dropped\n")
    for n, p in prov.items():
        print(f"  {n:34s} board {p['clean_board']:9.4f}  gated={p['gated']}  {p['source'][:60]}")
