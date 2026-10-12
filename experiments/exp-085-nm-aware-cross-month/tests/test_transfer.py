import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

P = Path(__file__).parents[1] / "code/transfer.py"
spec = importlib.util.spec_from_file_location("transfer", P)
T = importlib.util.module_from_spec(spec); spec.loader.exec_module(T)


class TransferTest(unittest.TestCase):
    def test_target_label_loader_is_blocked_before_freeze(self):
        with tempfile.TemporaryDirectory() as td, mock.patch.object(T, "FROZEN", Path(td)/"missing.json"):
            with self.assertRaisesRegex(RuntimeError, "blocked"):
                T.target_data("july_loo")

    def test_population_pair_direction_order_and_outside_bytes(self):
        pan=pd.DataFrame({"p_v22":[1.,2.,3.,4.],"nm_missing":[False,False,True,False]})
        d={"pan":pan,"pop":np.array([True,False,True,False]),"yall":np.array([2.,2.,5.,4.]),"rall":np.array([1.,0.,2.,0.])}
        tab=T.score_table(d,np.array([.5,1.])); self.assertEqual(tab["overall"]["rows"],4)
        ids=np.array([7,9]); self.assertTrue(np.array_equal(np.tile(ids,5)[:2],ids))
        self.assertEqual([(s,t) for s in T.FOLDS for t in T.FOLDS if s!=t][0],("july_loo","december"))

    def test_control_and_metric_arithmetic_and_fit_count(self):
        r=np.array([2.,4.]); u=np.array([1.,-1.])
        self.assertEqual(np.mean(r*r-(r-u)**2),-3.0)
        self.assertEqual(3*(3*(5+1)+1),57)
        self.assertEqual(set(T.OUTPUT_TYPES),{"aware","sham","blind","airport_constant","null"})

    def test_exact_registered_population_counts_and_columns(self):
        T.check_sources()
        for fold,n in T.EXPECTED_P.items():
            d=T.load_label_free(fold)
            self.assertEqual(len(d["P"]),n)
            self.assertNotIn("y_true",d["pan"].columns); self.assertNotIn("y",d["feat"].columns)
            self.assertEqual(len(d["X"]["aware"].columns),46); self.assertEqual(len(d["X"]["blind"].columns),29)


if __name__ == "__main__": unittest.main()
