"""Replace the ablation table in RESULTS.md with the four-slice x three-fold form.

COMMON.md: "Every RESULTS table must show RMSE overall, RMSE on rows with target
at or below 3600 s, RMSE on rows above, and RMSE on no-NM rows, for each fold."
"""
import json
import pandas as pd
import prc.data as d

R = d.ROOT / "experiments/exp-003-anatomy/RESULTS.md"
LABEL = {
 "A_base": "A base",
 "B_base+stats": "B base + StandStats",
 "N_base_nosameflight": "N base without the same-flight-arrival pair",
 "U_base+callsign_arrival": "U base + same-callsign-as-an-earlier-arrival",
 "M_base+occ_nbrdep": "M base + neighbouring departures at the stand",
 "O_base_target_taxi": "O base, target = taxi not the offset",
 "L_base+occ_nextarr": "L base + next arrival at the stand",
 "Q_base+phi": "Q base + phi",
 "D_base+nbr": "D base + neighbouring departures at the airport",
 "K_base+occ_prevarr": "K base + previous arrival at the stand",
 "C_base+occ": "C base + all stand-occupancy features",
 "E_base+stats+occ": "E base + StandStats + occupancy",
 "I_all_minus_nbr": "I everything minus airport neighbours",
 "G_all_minus_occ": "G everything minus stand occupancy",
 "S_best_phi_target": "S best, target = the phi residual",
 "R_best+phi": "R best + phi",
 "V_best+callsign_arrival": "V best + the same-callsign link",
 "H_all_minus_occb": "H everything minus parent-stand occupancy",
 "F_all": "F everything",
 "J_all_minus_stats": "J everything minus StandStats",
 "T_best": "T base + occupancy + airport neighbours (**recommended**)",
 "P_all_clip0": "P everything, predictions clipped at zero",
}


def load():
    v, feats = {}, {}
    for line in open(d.ROOT / "experiments/exp-003-anatomy/climb.log"):
        r = json.loads(line)
        n = r["name"].replace("_3fold", "")
        for f, s in r["res"].items():
            v.setdefault(n, {})[f] = s
            feats[n] = s["n_feat"]
    return v, feats


def main():
    v, feats = load()
    base = v["A_base"]["july"]["rmse_bulk"]
    order = sorted(LABEL, key=lambda n: -v[n]["july"]["rmse_bulk"])
    # rows whose feature set contains the `nbr` block were affected by the
    # +/-30-minute window bug; only T, D and F were re-measured after the fix.
    NBR_ROWS = {"D_base+nbr", "F_all", "G_all_minus_occ", "H_all_minus_occb",
                "J_all_minus_stats", "P_all_clip0", "R_best+phi",
                "S_best_phi_target", "T_best", "V_best+callsign_arrival"}
    REDONE = {"T_best", "D_base+nbr", "F_all"}
    out = ["| model | feat | fold | RMSE all | RMSE <=3600 | RMSE >3600 | RMSE no-NM | **RMSE bulk** | vs base (july bulk) | window fix |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for n in order:
        dlt = v[n]["july"]["rmse_bulk"] - base
        for i, f in enumerate(("july", "december", "january")):
            s = v[n][f]
            out.append(f"| {LABEL[n] if i == 0 else ''} | {feats[n] if i == 0 else ''} | {f} | "
                       f"{s['rmse_all']:.1f} | {s['rmse_le3600']:.1f} | {s['rmse_gt3600']:.0f} | "
                       f"{s['rmse_noNM']:.0f} | **{s['rmse_bulk']:.2f}** | "
                       f"{('–' if n == 'A_base' else f'{dlt:+.2f}') if i == 0 else ''} | "
                       f"{('' if i else ('after' if n in REDONE else ('**before**' if n in NBR_ROWS else 'n/a')))} |")
    text = "\n".join(out)
    s = R.read_text()
    start = s.index("| model | features | july | december | january | vs base (july) |")
    if "| model | feat | fold | RMSE all |" in s:
        start = s.index("| model | feat | fold | RMSE all |")
    end = s.index("Every row is on all three folds")
    R.write_text(s[:start] + text + "\n\n" + s[end:])
    print("inlined", len(order), "models x 3 folds")


if __name__ == "__main__":
    main()
