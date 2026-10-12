"""Re-run every row whose feature set contains the `nbr` block, after the
+/-30-minute window bug in `nbr_apt_30m` was fixed (2026-09-05)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate as A
import prc.data as d

A.PARAMS["num_threads"] = 12
LOG = d.ROOT / "experiments/exp-003-anatomy/climb.log"
F3 = ("july", "december", "january")
ALL = ["base", "stats", "occ", "occb", "nbr"]
BEST = ["base", "occ", "nbr"]
A.run("T_best", BEST, folds=F3, save_oof="T_best", log=LOG)
A.run("D_base+nbr", ["base", "nbr"], folds=F3, log=LOG)
A.run("F_all", ALL, folds=F3, save_oof="F_all", log=LOG)
A.run("J_all_minus_stats", ["base", "occ", "occb", "nbr"], folds=F3, log=LOG)
A.run("G_all_minus_occ", ["base", "stats", "occb", "nbr"], folds=F3, log=LOG)
A.run("H_all_minus_occb", ["base", "stats", "occ", "nbr"], folds=F3, log=LOG)
A.run("P_all_clip0", ALL, folds=F3, log=LOG, clip0=True)
A.run("R_best+phi", BEST + ["phi"], folds=F3, save_oof="R_best_phi", log=LOG)
A.run("S_best_phi_target", BEST + ["phi"], folds=F3, log=LOG, target="phi_resid")
A.run("V_best+callsign_arrival", BEST + ["csn"], folds=F3, log=LOG)
