import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ablate as A
import prc.data as d
LOG = d.ROOT / "experiments/exp-003-anatomy/climb.log"
A.run("U_base+callsign_arrival", ["base", "csn"], folds=("july", "december"), log=LOG)
A.run("V_best+callsign_arrival", ["base", "occ", "nbr", "csn"], folds=("july", "december"), log=LOG)
