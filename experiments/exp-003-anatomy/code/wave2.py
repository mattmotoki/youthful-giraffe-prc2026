import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate as A
import prc.data as d

LOG = d.ROOT / "experiments/exp-003-anatomy/climb.log"
ALL = ["base", "stats", "occ", "occb", "nbr"]
plan = [
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
]
for name, blocks, kw in plan:
    A.run(name, blocks, log=LOG, **kw)
# three-fold rows for the two reference models, with OOF for the full set
A.run("A_base_3fold", ["base"], folds=("july", "december", "january"), log=LOG)
A.run("F_all_3fold", ALL, folds=("july", "december", "january"), save_oof="F_all", log=LOG)
