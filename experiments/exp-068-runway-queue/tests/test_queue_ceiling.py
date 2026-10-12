from __future__ import annotations
import importlib.util
from pathlib import Path
import unittest
import numpy as np
import pandas as pd

P = Path(__file__).parents[1] / "code" / "queue_ceiling.py"
spec = importlib.util.spec_from_file_location("queue_ceiling", P)
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)

def frame():
    return pd.DataFrame({"MVT_ID_mvt": [3., 1., 2.], "FLIGHT_ID_mvt": [1., 2., 3.], "FLIGHT_mvt": ["X1"]*3,
        "PHASE_mvt": ["DEP"]*3, "ADEP_mvt": ["AAA"]*3, "ADES_mvt": ["BBB"]*3,
        "MVT_TIME_UTC_mvt": pd.to_datetime(["2026-01-01 01:00Z"]*3), "SCHED_TIME_UTC_mvt": pd.to_datetime(["2026-01-01 00:00Z"]*3),
        "LOBT_flt": pd.to_datetime(["2026-01-01 00:07Z"]*3), "EOBT_1_flt": pd.NaT, "IOBT_flt": pd.NaT,
        "RUNWAY_mvt": ["R1"]*3, "STAND_mvt": ["S1"]*3, "AIRCRAFT_TYPE_mvt": ["A320"]*3,
        "ADEP_flt": ["AAA"]*3, "ADES_flt": ["BBB"]*3, "AIRCRAFT_TYPE_flt": ["A320"]*3,
        "WK_TBL_CAT_flt": ["M"]*3, "AIRCRAFT_OPERATOR_flt": ["OP"]*3,
        "BLOCK_TIME_UTC_mvt": pd.to_datetime(["2026-01-01 00:10Z"]*3), "TAXITIME_SEC_mvt": [300,301,302],
        "y_true": [300,301,302], "y_pred": [290,291,292], "label_derived__residual": [10,10,10]})

class QueueCeilingTests(unittest.TestCase):
    def test_phase_a_removes_all_label_side_columns(self):
        d = q.planned_departures(frame(), "synthetic")
        self.assertFalse(q.FORBIDDEN.intersection(d.columns)); self.assertNotIn("MVT_TIME_UTC_mvt", q.INPUT_COLUMNS)

    def test_all_census_bytes_ignore_one_synthetic_label_mutation(self):
        base = q.census_outputs([("synthetic", frame())]); changed = frame()
        for c in q.FORBIDDEN: changed[c] = np.arange(3, dtype=float) + 987654
        changed["label_derived__any_formula"] = [-999,0,999]
        self.assertEqual(base.keys(), changed.pipe(lambda x: q.census_outputs([("synthetic", x)])).keys())
        got = q.census_outputs([("synthetic", changed)])
        for name in ("slot-census.csv", "source-summary.csv", "airport-summary.csv", "census.json"):
            self.assertEqual(base[name], got[name], name)

    def test_cutoff_cohort_and_self_excluded_demand(self):
        d = q.planned_departures(frame(), "synthetic")
        self.assertTrue(d.plan_source.eq("LOBT").all()); self.assertEqual(q.self_excluded_slot_demand(d).tolist(), [2,2,2])

    def test_self_excluded_offsets_and_strict_earlier_month(self):
        h = pd.DataFrame({"row_id": range(1,8), "month": ["2025-01","2025-01","2025-02","2025-02","2025-03","2025-03","2025-04"],
            "airport": ["A"]*7, "operator": ["O"]*7, "offset": [10,30,900,900,900,900,9999]})
        query = pd.Series({"row_id": 1, "airport":"A", "operator":"O"})
        value, level, support = q.resolve_prior(h, query, "offset", [["airport","operator"],["airport"]], "2025-02", 1, q.PLAN_OFFSET_CLIP)
        self.assertEqual((value, level, support), (30.0, "airportxoperator", 1))
        with self.assertRaises(ValueError): q.resolve_prior(h, query, "offset", [["airport"]], "2025-01", 1)

    def test_plan_offset_every_registered_fallback(self):
        levels=[["airport","operator","flight","destination"],["airport","operator"],["airport"]]
        h=pd.DataFrame({"row_id":range(8),"month":["2025-01"]*8,
            "airport":["A","A","A","A","B","B","C","D"],"operator":["O","O","O","O","P","P","Q","R"],
            "flight":["F","F","G","G","H","H","I","J"],"destination":["X","X","Y","Y","Z","Z","W","V"],
            "offset":[10,30,50,70,90,110,130,150]})
        cases=[({"airport":"A","operator":"O","flight":"F","destination":"X"},(20.,"airportxoperatorxflightxdestination")),
            ({"airport":"A","operator":"O","flight":"Q","destination":"Q"},(40.,"airportxoperator")),
            ({"airport":"B","operator":"Q","flight":"Q","destination":"Q"},(100.,"airport")),
            ({"airport":"Q","operator":"Q","flight":"Q","destination":"Q"},(80.,"global"))]
        for fields,expected in cases:
            value,level,_=q.resolve_prior(h,pd.Series({"row_id":99,**fields}),"offset",levels,"2025-02",2,q.PLAN_OFFSET_CLIP)
            self.assertEqual((value,level),expected)

    def test_unimpeded_prior_every_sparse_fallback_and_clip(self):
        levels = [["airport","stand_zone","runway","wake_type","quarter_hour"], ["airport","runway","wake_type"], ["airport","runway"], ["airport"]]
        quarter=pd.Timestamp("2025-01-01 00:00Z")
        h=pd.DataFrame({"row_id":range(10),"month":["2025-01"]*10,
            "airport":["A","A","A","A","B","B","C","C","D","E"], "stand_zone":["S","S","X","X","Y","Y","Z","Z","W","V"],
            "runway":["R","R","R","R","Q","Q","T","T","U","V"], "wake_type":["M","M","M","M","H","H","L","L","M","H"],
            "quarter_hour":[quarter]*10, "u":[10,110,200,220,300,320,400,420,500,4000]})
        cases=[({"airport":"A","stand_zone":"S","runway":"R","wake_type":"M","quarter_hour":quarter},(60.,"airportxstand_zonexrunwayxwake_typexquarter_hour")),
            ({"airport":"A","stand_zone":"Q","runway":"R","wake_type":"M","quarter_hour":quarter},(155.,"airportxrunwayxwake_type")),
            ({"airport":"C","stand_zone":"Q","runway":"T","wake_type":"X","quarter_hour":quarter},(410.,"airportxrunway")),
            ({"airport":"B","stand_zone":"Q","runway":"X","wake_type":"X","quarter_hour":quarter},(310.,"airport")),
            ({"airport":"Q","stand_zone":"Q","runway":"X","wake_type":"X","quarter_hour":quarter},(310.,"global"))]
        for fields, expected in cases:
            query=pd.Series({"row_id":99,**fields})
            value,level,_=q.resolve_prior(h,query,"u",levels,"2025-02",2,q.UNIMPEDED_CLIP)
            self.assertEqual((value,level),expected)

    def test_registered_plan_and_low_demand_targets(self):
        raw=pd.DataFrame({"row_id":[1,2,3],"airport":["A"]*3,"runway":["R"]*3,"stand_zone":["S"]*3,
            "WK_TBL_CAT_flt":["M"]*3,"AIRCRAFT_TYPE_mvt":["A320"]*3,
            "plan_time":pd.to_datetime(["2025-01-01 00:00Z"]*3),
            "block_time":pd.to_datetime(["2025-01-01 00:01Z","2025-01-01 00:05Z","2025-01-01 00:20Z"]),
            "event_time":pd.to_datetime(["2025-01-01 00:06Z","2025-01-01 00:10Z","2025-01-01 00:27Z"])})
        offsets=q.historical_plan_offsets(raw); self.assertEqual(offsets.plan_offset_seconds.tolist(),[60.,300.,1200.])
        low=q.historical_unimpeded_events(raw)
        self.assertEqual(low.row_id.tolist(),[3]); self.assertEqual(low.unimpeded_seconds.tolist(),[420.])

    def test_month_forward_service_day_boundary_and_fallbacks(self):
        events = pd.DataFrame({"row_id":[1,2,3], "airport":["A"]*3, "runway":["R"]*3,
            "event_time":pd.to_datetime(["2025-01-01 23:59Z","2025-01-02 00:01Z","2025-01-02 00:02Z"])})
        got = q.historical_service_events(events)
        self.assertEqual(got.row_id.tolist(),[3]); self.assertEqual(got.service_interval_seconds.tolist(),[60.])
        equal=pd.DataFrame({"row_id":[1,2,3,4],"airport":["A"]*4,"runway":["R"]*4,
            "event_time":pd.to_datetime(["2025-01-01 00:00Z","2025-01-01 00:00Z","2025-01-01 00:01Z","2025-01-02 00:00Z"])})
        self.assertEqual(q.historical_service_events(equal).service_interval_seconds.tolist(),[60.])
        service={"min_support":3,("A","R1"):(5,3),("A","*"):(35,4),("*","*"):(999,9)}
        self.assertEqual(q.resolve_service_seconds("R1","A",service),(20.,"runway")); self.assertEqual(q.resolve_service_seconds("RX","A",service),(35.,"airport")); self.assertEqual(q.resolve_service_seconds("R","B",service),(180.,"global"))

    def test_queue_identity_ties_independent_runways_and_conservation(self):
        jobs=pd.DataFrame({"row_id":[3,1,2,4],"airport":["A"]*4,"runway":["R","R","R","Q"],"release_time":[0.]*4,"unimpeded_seconds":[100.]*4})
        service={"min_support":2,("A","R"):(30.,4),("A","Q"):(30.,4),("*","*"):(50.,99)}
        got=q.simulate_runway_queue(jobs,service)
        self.assertEqual(got.row_id.tolist(),[1,2,3,4]); self.assertEqual(got.queue_arrival_time.tolist(),[100.]*4)
        self.assertEqual(got.queue_wait_seconds.tolist(),[0.,30.,60.,0.]); self.assertEqual(got.taxi_prediction_seconds.tolist(),[100.,130.,160.,100.])
        self.assertTrue(np.array_equal((got.service_finish-got.service_start).to_numpy(),got.service_seconds.to_numpy()))

    def test_oracle_uses_true_block_order_only(self):
        f=pd.DataFrame({"row_id":[1,2,3],"airport":["A"]*3,"runway":["R"]*3,"utc_day":["d"]*3,
            "deployable_arrival_time":[30.,10.,20.],"true_block_time":[100.,200.,300.]})
        first=q.oracle_order_only_arrivals(f); f["true_block_time"]=[-900.,0.,999999.]
        second=q.oracle_order_only_arrivals(f)
        self.assertEqual(first.to_numpy().tobytes(),second.to_numpy().tobytes())
        jobs=pd.DataFrame({"row_id":f.row_id,"airport":f.airport,"runway":f.runway,"unimpeded_seconds":[5.]*3,"queue_arrival_time":first})
        service={"min_support":1,("A","R"):(10.,3)}
        p1=q.simulate_runway_queue(jobs,service).taxi_prediction_seconds.to_numpy()
        jobs["queue_arrival_time"]=second
        p2=q.simulate_runway_queue(jobs,service).taxi_prediction_seconds.to_numpy()
        self.assertEqual(p1.tobytes(),p2.tobytes())

    def test_query_removal_rebuilds_neighbor_targets(self):
        raw=pd.DataFrame({"row_id":[1,2,3],"month":["2025-01"]*3,"airport":["A"]*3,"runway":["R"]*3,"stand_zone":["S"]*3,
            "WK_TBL_CAT_flt":["M"]*3,"AIRCRAFT_TYPE_mvt":["A320"]*3,"plan_time":pd.to_datetime(["2025-01-01 00:00Z"]*3),
            "block_time":pd.to_datetime(["2025-01-01 00:00Z","2025-01-01 00:05Z","2025-01-01 00:14Z"]),
            "event_time":pd.to_datetime(["2025-01-01 01:00Z","2025-01-01 01:01Z","2025-01-01 01:03Z"])})
        self.assertEqual(q.historical_unimpeded_events(raw).row_id.tolist(),[])
        self.assertEqual(q.historical_unimpeded_events(raw,2).row_id.tolist(),[1,3])
        self.assertEqual(q.historical_service_events(raw).set_index("row_id").loc[3,"service_interval_seconds"],120.)
        self.assertEqual(q.historical_service_events(raw,2).set_index("row_id").loc[3,"service_interval_seconds"],180.)

    def test_frozen_unimpeded_category_and_anchor(self):
        h=pd.DataFrame({"WK_TBL_CAT_flt":["H",None,None],"AIRCRAFT_TYPE_mvt":["A","B",None],
            "plan_time":pd.to_datetime(["2025-01-01 00:14Z","2025-01-01 00:15Z","2025-01-01 00:29Z"])})
        got=q.freeze_unimpeded_inputs(h)
        self.assertEqual(got.wake_type.tolist(),["H","B","__MISSING__"])
        self.assertEqual(got.quarter_hour.tolist(),list(pd.to_datetime(["2025-01-01 00:00Z","2025-01-01 00:15Z","2025-01-01 00:15Z"])))

    def test_alignment_rejects_dtype_order_missing_and_duplicates(self):
        q.assert_exact_row_alignment(np.array([1,2,3],dtype="float64"),np.array([1,2,3],dtype="float64"))
        bad=[np.array([1,2,3],dtype="int64"),np.array([2,1,3],dtype="float64"),np.array([1,2],dtype="float64"),np.array([1,2,2],dtype="float64")]
        for ids in bad:
            with self.assertRaises(AssertionError): q.assert_exact_row_alignment(np.array([1,2,3],dtype="float64"),ids)

    def test_checked_in_output_schemas_are_label_blind(self):
        for name in ("slot-census.csv","source-summary.csv","airport-summary.csv"):
            cols=pd.read_csv(P.parents[1]/"analysis"/name,nrows=0).columns
            self.assertFalse(q.FORBIDDEN.intersection(cols)); self.assertFalse(any(c.startswith("label_derived__") for c in cols))

if __name__ == "__main__": unittest.main()
