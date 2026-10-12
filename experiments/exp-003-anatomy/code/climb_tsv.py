"""Render climb.log (JSON, one line per model x fold) into the COMMON.md climb format.

Columns: iteration | mechanism changed | july RMSE | december RMSE | fraction of
the remaining gap closed. RMSEs are the bulk metric (NM join, taxi <= 3600 s).
The gap is measured from the reference naive predictor (stand-median offset,
clipped at zero: 376.6 july, 320.0 december) to the estimated bulk ceiling of
200 s (RESULTS.md finding 9), so the fraction is an estimate with a stated
denominator, not a measurement.
"""
import json
import prc.data as d

BASE = {"july": 376.6, "december": 320.0, "january": 308.6}
CEIL = 200.0
# One line per model: a model run again to add a fold updates that model's line,
# it does not start a new one. `X_3fold` is the same model as `X`.
merged = {}
for line in open(d.ROOT / "experiments/exp-003-anatomy/climb.log"):
    r = json.loads(line)
    name = r["name"].replace("_3fold", "")
    mech = "+".join(r["blocks"]) + ("" if r.get("target", "offset") == "offset"
                                    else f" [target={r['target']}]")
    e = merged.setdefault(name, {"mech": mech, "folds": {}})
    e["mech"] = mech
    for f, v in r["res"].items():
        e["folds"][f] = v["rmse_bulk"]
rows = [(n, e["mech"], e["folds"].get("july"), e["folds"].get("december"),
         e["folds"].get("january")) for n, e in merged.items()]
out = [ "iteration\tmechanism changed\tjuly bulk RMSE\tdecember bulk RMSE\tjanuary bulk RMSE\t"
        "fraction of the gap to a 200 s ceiling closed (july)" ]
for i, (name, mech, j, dec, jan) in enumerate(rows, 1):
    frac = "" if j is None else f"{(BASE['july'] - j) / (BASE['july'] - CEIL):.3f}"
    f = lambda v: "" if v is None else f"{v:.2f}"
    out.append(f"{i}\t{name}: {mech}\t{f(j)}\t{f(dec)}\t{f(jan)}\t{frac}")
p = d.ROOT / "experiments/exp-003-anatomy/climb.tsv"
p.write_text("\n".join(out) + "\n")
print("\n".join(out))
