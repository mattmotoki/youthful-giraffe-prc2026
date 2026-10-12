from __future__ import annotations
import importlib.util, json, os, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

P = Path(__file__).with_name("screen.py")
sp = importlib.util.spec_from_file_location("exp091", P); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
T0 = pd.Timestamp("2026-01-01T00:00:00Z")

def row(i, phase="DEP", t=T0, sched=None, flight="AB1", adep="AAA", ades="BBB", callsign="CS1", runway="R1", nm=True):
    d = {"MVT_ID_mvt": float(i), "FLIGHT_mvt": flight, "PHASE_mvt": phase, "ADEP_mvt": adep,
         "ADES_mvt": ades, "MVT_TIME_UTC_mvt": t, "SCHED_TIME_UTC_mvt": sched or t,
         "RUNWAY_mvt": runway, "AIRCRAFT_TYPE_mvt": "A", **{x: ("x" if nm else None) for x in m.NM_FIELDS}}
    d["CALLSIGN_flt"] = callsign if nm else None
    return d

def frame(rows): return pd.DataFrame(rows)

def test_s2_edges_midnight_missing_collision_and_controls():
    d = frame([row(1,t=T0,flight="AB1",callsign="C1"), row(2,t=T0+pd.Timedelta(seconds=60),flight="EDGE",callsign="EDGE"),
               row(3,t=T0+pd.Timedelta(seconds=61),flight="AB1",callsign="C1"),
               row(4,t=T0+pd.Timedelta(hours=23,seconds=59),flight="Q",callsign=None),
               row(5,t=T0+pd.Timedelta(days=1),flight="Q",callsign=None)])
    x=m.s2(d); assert x["lanes"]["service_designator"]["supported"] == 1
    assert x["intersection"] == 1 and x["controls"]["service_designator"]["previous_utc_day"] == 1
    c=frame([row(1,flight="ＡB1"),row(2,t=T0+pd.Timedelta(minutes=2),flight="AB1")])
    assert m.s2(c)["normalized_collision_groups"]["service_designator"] == 1

def test_s2_max_edge_and_ambiguity():
    d=frame([row(1,t=T0),row(2,t=T0+pd.Timedelta(seconds=86399)),row(3,t=T0+pd.Timedelta(seconds=500))])
    assert m.s2(d)["lanes"]["service_designator"]["supported"] == 0

def test_s3_direction_window_ambiguity_controls_and_crossing():
    dep=row(1,t=T0+pd.Timedelta(minutes=10),sched=T0-pd.Timedelta(minutes=10),nm=False)
    good=row(2,phase="ARR",t=T0+pd.Timedelta(minutes=15),adep="AAA",ades="BBB")
    wrong=row(3,phase="ARR",t=T0+pd.Timedelta(minutes=15),adep="BBB",ades="AAA")
    x=m.s3(frame([dep,good,wrong])); assert x["crossing_population"] == 1 and x["supported"] == 1
    dup=row(4,phase="ARR",t=T0+pd.Timedelta(minutes=16),adep="AAA",ades="BBB")
    assert m.s3(frame([dep,good,dup]))["ambiguity_rejections"] == 1
    edge=row(5,phase="ARR",t=T0+pd.Timedelta(minutes=15),adep="AAA",ades="BBB")
    assert m.s3(frame([dep,edge]))["supported"] == 1
    late=row(6,phase="ARR",t=dep["MVT_TIME_UTC_mvt"]+pd.Timedelta(hours=18,seconds=1),adep="AAA",ades="BBB")
    assert m.s3(frame([dep,late]))["supported"] == 0

def test_month_and_leap_boundaries_zero_support():
    d=frame([row(1,t=pd.Timestamp("2026-01-01T00:05:00Z")), row(2,t=pd.Timestamp("2026-01-01T00:04:59Z")),
             row(3,t=pd.Timestamp("2026-07-01T00:59:59Z")), row(4,t=pd.Timestamp("2026-01-02T00:00:00Z")),
             row(5,t=pd.Timestamp("2024-02-29T00:00:00Z"))])
    x=m.s5(d); assert x["months"]["january"]["prefix_minutes"]["5"]["total"] == 1
    assert x["months"]["july"]["prefix_minutes"]["60"]["total"] == 1
    assert x["months"]["january"]["later_day_first_hour_control"]["total"] == 1

def test_s4_order_thresholds_missing_runway_nonfinite_and_negative():
    d=frame([row(2,runway=None),row(1)])
    p=pd.DataFrame({"MVT_ID_mvt":[1.,2.],"TAXITIME_SEC_mvt":[119.999,np.nan]})
    x=m.s4(d,p); assert x["below_seconds"] == {"120":1,"180":1,"240":1}
    assert x["refusal_checks"]["non_finite"] == 1
    p.loc[1,"TAXITIME_SEC_mvt"]=-1; assert m.s4(d,p)["refusal_checks"]["negative"] == 1

def test_duplicate_nonfinite_unmatched_ids_rejected():
    d=frame([row(1),row(2)]); p=pd.DataFrame({"MVT_ID_mvt":[1.,1.],"TAXITIME_SEC_mvt":[1.,2.]})
    for q in [p, pd.DataFrame({"MVT_ID_mvt":[1.,np.nan],"TAXITIME_SEC_mvt":[1.,2.]}),
              pd.DataFrame({"MVT_ID_mvt":[1.,3.],"TAXITIME_SEC_mvt":[1.,2.]})]:
        try: m.s4(d,q); assert False
        except ValueError: pass

def test_source_and_schema_mutation_and_forbidden_projection():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); p=root/"x.parquet"; pq.write_table(pa.table({"a":[1]}),p)
        spec={"path":"x.parquet","size_bytes":p.stat().st_size,"sha256":m.sha256_file(p),"rows":1,
              "schema":m.schema_pairs(p),"projected_columns":["a"]}
        assert m.validate_source(root,spec)==p
        p.write_bytes(p.read_bytes()+b"x")
        try: m.validate_source(root,spec); assert False
        except ValueError: pass
        q=root/"ranking.parquet"; pq.write_table(pa.table({"TAXITIME_SEC_mvt":[1]}),q)
        bad={"path":"ranking.parquet","size_bytes":q.stat().st_size,"sha256":m.sha256_file(q),"rows":1,
             "schema":m.schema_pairs(q),"projected_columns":["TAXITIME_SEC_mvt"]}
        try: m.validate_source(root,bad); assert False
        except ValueError: pass

def test_aggregate_only_standard_json_and_atomic_replacement():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"x.json"; p.write_text("old")
        m.atomic_json(p,{"count":0,"control":{"n":1}}); assert json.loads(p.read_text())["count"]==0
        assert not list(Path(td).glob(".x.json.*"))
        for bad in [{"MVT_ID_mvt":1},{"x":float("nan")},{"x":"CALLSIGN_flt"},{"x":"CS1"},
                    {"event_time":"2026-01-01T00:00:00Z"},{"x":"2026-01-01T00:00:00Z"}]:
            try: m.atomic_json(p,bad); assert False
            except ValueError: pass

def test_complete_result_uses_exact_writer_path():
    dep=frame([row(1,t=T0),row(2,t=T0+pd.Timedelta(minutes=2))])
    pred=pd.DataFrame({"MVT_ID_mvt":[1.,2.],"TAXITIME_SEC_mvt":[120.,180.]})
    screens={"S2":m.s2(dep),"S3":m.s3(dep),"S4":m.s4(dep,pred),"S5":m.s5(dep)}
    result={"format_version":1,
            "claim_boundary":"input-only support; no prediction-value or physical-validity claim",
            "source_sha256":{"data/raw/ranking.parquet":"0"*64,
                             "submissions/uploaded/youthful-giraffe_v22.parquet":"1"*64},
            "screens":screens,"screens_clearing_gate":[k for k,v in screens.items() if v["gate_pass"]]}
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/"aggregate.json"; p.write_text("old")
        m.atomic_json(p,result)
        raw=p.read_text(); assert json.loads(raw)==result and "NaN" not in raw
        assert not list(Path(td).glob(".aggregate.json.*"))

def test_deterministic_permutation_and_query_removal():
    d=frame([row(1,flight="A"),row(2,t=T0+pd.Timedelta(minutes=2),flight="B")])
    key=m.normalized(d,"FLIGHT_mvt"); a=m.permute_within(d,key,["ADEP_mvt"]); b=m.permute_within(d,key,["ADEP_mvt"])
    assert a.tolist()==b.tolist()==["B","A"]
    # A removed candidate cannot survive a recomputed reciprocal match.
    group=pd.Series([("x",), ("x",)],index=d.index)
    assert m.reciprocal_pairs(d.MVT_TIME_UTC_mvt,group,60,86399)=={0}
    assert m.reciprocal_pairs(d.iloc[:1].MVT_TIME_UTC_mvt,group.iloc[:1],60,86399)==set()

def test_missing_identity_and_exact_18h_edge():
    dep=row(1,t=T0+pd.Timedelta(minutes=1),sched=T0-pd.Timedelta(minutes=1),nm=False)
    arr=row(2,phase="ARR",t=dep["MVT_TIME_UTC_mvt"]+pd.Timedelta(hours=18),adep="AAA",ades="BBB")
    assert m.s3(frame([dep,arr]))["supported"]==1
    arr["FLIGHT_mvt"]=None; assert m.s3(frame([dep,arr]))["supported"]==0

def test_s3_previous_day_branch_and_normalized_collision():
    dep=row(1,t=T0+pd.Timedelta(minutes=1),sched=T0-pd.Timedelta(minutes=1),nm=False)
    arr=row(2,phase="ARR",t=T0-pd.Timedelta(hours=1),adep="AAA",ades="BBB")
    x=m.s3(frame([dep,arr])); assert x["supported"]==1 and x["arrival_date_split"]["D_minus_1"]==1
    arr2=row(3,phase="ARR",t=T0-pd.Timedelta(hours=2),flight="ＡB1",adep="AAA",ades="BBB")
    assert m.s3(frame([dep,arr,arr2]))["supported"]==0

if __name__ == "__main__":
    tests=[v for k,v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests: t()
    print(f"{len(tests)} tests passed")
