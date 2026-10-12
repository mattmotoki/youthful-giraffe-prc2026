import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate as A
import prc.data as d

LOG = d.ROOT / "experiments/exp-003-anatomy/climb.log"
ALL = ["base", "occ", "nbr"]          # `stats` and `occb` measured out in wave 2
F3 = ("july", "december", "january")
A.run("Q_base+phi", ["base", "phi"], folds=F3, log=LOG)
A.run("R_best+phi", ALL + ["phi"], folds=F3, save_oof="R_best_phi", log=LOG)
A.run("S_best_phi_target", ALL + ["phi"], folds=F3, log=LOG, target="phi_resid")
A.run("T_best", ALL, folds=F3, save_oof="T_best", log=LOG)
