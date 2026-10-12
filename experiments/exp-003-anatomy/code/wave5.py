"""Backfill the january fold for every ablation row, and re-run the two callsign
rows after the `cs_prev_arr_gap` bug fix (Sol round 2, findings 3 and 4)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate as A
import prc.data as d

A.PARAMS["num_threads"] = 12
LOG = d.ROOT / "experiments/exp-003-anatomy/climb.log"
ALL = ["base", "stats", "occ", "occb", "nbr"]
BEST = ["base", "occ", "nbr"]

# the two callsign rows, all three folds, with the fixed features
A.run("U_base+callsign_arrival", ["base", "csn"], folds=("july", "december", "january"), log=LOG)
A.run("V_best+callsign_arrival", BEST + ["csn"], folds=("july", "december", "january"), log=LOG)

# january backfill for every row that only has july and december
JAN = ("january",)
for name, blocks, kw in [
    ("B_base+stats", ["base", "stats"], {}),
    ("C_base+occ", ["base", "occ"], {}),
    ("D_base+nbr", ["base", "nbr"], {}),
    ("E_base+stats+occ", ["base", "stats", "occ"], {}),
    ("G_all_minus_occ", [b for b in ALL if b != "occ"], {}),
    ("H_all_minus_occb", [b for b in ALL if b != "occb"], {}),
    ("I_all_minus_nbr", [b for b in ALL if b != "nbr"], {}),
    ("J_all_minus_stats", [b for b in ALL if b != "stats"], {}),
    ("K_base+occ_prevarr", ["base", "occ_prevarr"], {}),
    ("L_base+occ_nextarr", ["base", "occ_nextarr"], {}),
    ("M_base+occ_nbrdep", ["base", "occ_nbrdep"], {}),
    ("N_base_nosameflight", ["base_nosf"], {}),
    ("O_base_target_taxi", ["base"], {"target": "taxi"}),
    ("P_all_clip0", ALL, {"clip0": True}),
    ("A_base", ["base"], {}),
    ("F_all", ALL, {}),
]:
    A.run(name, blocks, folds=JAN, log=LOG, **kw)
