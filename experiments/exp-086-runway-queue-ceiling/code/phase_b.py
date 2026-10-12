"""Complete exp-086 measurement path. Real entry points are locked until desk acceptance."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, re, subprocess
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
_spec = importlib.util.spec_from_file_location("exp086_queue", HERE / "queue_ceiling.py")
q = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(q)
ACCEPTANCE = EXP / "review-acceptance.json"
MANIFEST = EXP / "source-manifest.json"
BINDING = EXP / "analysis" / "comparator-binding.json"
FOLDS = {"july_loo": "2025-07", "december": "2025-12", "january": "2025-01"}
LABEL_COLUMNS = list(q.INPUT_COLUMNS) + ["BLOCK_TIME_UTC_mvt", "MVT_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]

def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())

def require_phase_b_authorization() -> dict:
    """The only Phase-B lock; call before any label-bearing source is parsed."""
    if not ACCEPTANCE.is_file(): raise PermissionError("Phase B has no desk acceptance")
    a = _load_json(ACCEPTANCE)
    required = {"experiment": "exp-086-runway-queue-ceiling", "registration_commit": q.REGISTRATION_COMMIT,
        "review_sha256": q.REVIEW_SHA256, "review_verdict": "VERDICT: BLOCK",
        "accepted_findings": [1,2,3,4,5,7,8,9], "rejected_finding": 6,
        "rejected_finding_citation": "assignments/runway-queue-ceiling-successor.md",
        "desk_authorization": "AUTHORIZE PHASE B"}
    for k, v in required.items():
        if a.get(k) != v: raise PermissionError("invalid Phase-B acceptance")
    repair = a.get("repair_commit")
    if not isinstance(repair, str) or not re.fullmatch(r"[0-9a-f]{40}", repair):
        raise PermissionError("invalid repair commit")
    check=subprocess.run(["git","cat-file","-e",repair+"^{commit}"],cwd=ROOT,capture_output=True)
    if check.returncode: raise PermissionError("repair commit does not exist")
    if q.sha256_file(EXP / "review" / "round1.md") != q.REVIEW_SHA256:
        raise PermissionError("review hash mismatch")
    return a

def _ordered_id_record(path: Path) -> dict:
    ids = pd.read_parquet(path, columns=["MVT_ID_mvt"])["MVT_ID_mvt"].to_numpy()
    dtype, raw = q.canonical_id_bytes(ids)
    return {"row_count": len(ids), "id_dtype": dtype,
            "ordered_id_sha256": hashlib.sha256(raw).hexdigest(), "unique_ids": len(np.unique(ids)) == len(ids)}

def phase_b_preflight() -> dict:
    """Verify all immutable bindings before reading any label column."""
    acceptance = require_phase_b_authorization()
    from prc.data import data_dir, month_file
    m = _load_json(MANIFEST); b = _load_json(BINDING)
    if m.get("frozen_census") != q.FROZEN_CENSUS: raise RuntimeError("census binding mismatch")
    if m.get("control_seeds") != q.CONTROL_SEEDS or m.get("bootstrap_draws") != 2000: raise RuntimeError("seed binding mismatch")
    if tuple(m.get("output_columns", q.OUTPUT_COLUMNS)) != q.OUTPUT_COLUMNS: raise RuntimeError("output schema mismatch")
    for month, expected in m["phase_b_source_hashes"].items():
        if q.sha256_file(month_file(month)) != expected: raise RuntimeError("Phase-B source hash mismatch")
    # Projection verification reads only the registered label-free columns.
    for month in ("2025-01", "2025-07", "2025-12"):
        f = pd.read_parquet(month_file(month), columns=list(q.INPUT_COLUMNS))
        if q.canonical_frame_hash(f) != m["projection_hashes"][month]: raise RuntimeError("projection hash mismatch")
    ranking=pd.read_parquet(data_dir()/"ranking.parquet",columns=list(q.INPUT_COLUMNS))
    if q.canonical_frame_hash(ranking) != m["projection_hashes"]["ranking"]: raise RuntimeError("ranking projection hash mismatch")
    matched = 0
    for month in m["phase_b_source_hashes"]:
        ids = pd.read_parquet(month_file(month), columns=["MVT_ID_mvt"])["MVT_ID_mvt"]
        matched += int(q.exclusion_mask(ids).sum())
    if matched != 2: raise RuntimeError("exclusion count mismatch")
    expected_weights = {"july_loo": .5571321275602379, "january": .4428678724397621, "december": 0.0}
    if {k:b["board_weights"][k] for k in expected_weights} != expected_weights: raise RuntimeError("weight mismatch")
    for rel, expected in b["mask_definition_sources"].items():
        if q.sha256_file(ROOT / rel) != expected: raise RuntimeError("mask source mismatch")
    for fold, record in b["folds"].items():
        path = ROOT / record["path"]
        if q.sha256_file(path) != record["file_sha256"]: raise RuntimeError("comparator hash mismatch")
        got = _ordered_id_record(path)
        for key in ("row_count", "id_dtype", "ordered_id_sha256", "unique_ids"):
            if got[key] != record[key]: raise RuntimeError("comparator ID binding mismatch")
    return {"acceptance": acceptance, "exclusion_matches": matched}

def _plan_time(d: pd.DataFrame) -> pd.Series:
    for c in ("LOBT_flt","EOBT_1_flt","IOBT_flt","SCHED_TIME_UTC_mvt"):
        d[c] = pd.to_datetime(d[c], utc=True, errors="coerce")
    return d.LOBT_flt.combine_first(d.EOBT_1_flt).combine_first(d.IOBT_flt).combine_first(d.SCHED_TIME_UTC_mvt)

def prepare_raw(frame: pd.DataFrame, month: str) -> pd.DataFrame:
    d = frame.loc[frame.PHASE_mvt.eq("DEP")].copy().reset_index(drop=True)
    d["row_id"] = d.MVT_ID_mvt; d["month"] = month; d["airport"] = d.ADEP_mvt
    d["runway"] = d.RUNWAY_mvt; d["supplied_stand"] = d.STAND_mvt.fillna("__MISSING__").astype(str)
    d["operator"] = d.AIRCRAFT_OPERATOR_flt; d["flight"] = d.FLIGHT_mvt; d["destination"] = d.ADES_mvt
    d["plan_time"] = _plan_time(d); d["block_time"] = pd.to_datetime(d.BLOCK_TIME_UTC_mvt, utc=True)
    d["event_time"] = pd.to_datetime(d.MVT_TIME_UTC_mvt, utc=True)
    return q.freeze_unimpeded_inputs(d)

def _resolve_vector(history: pd.DataFrame, queries: pd.DataFrame, target: str, levels: list[list[str]], clip) -> tuple[np.ndarray,np.ndarray]:
    values=np.full(len(queries),np.nan); supported=np.zeros(len(queries),bool)
    for cols in levels + [[]]:
        unresolved = ~supported
        if not unresolved.any(): break
        if cols:
            agg=history.groupby(cols,dropna=False)[target].agg(["median","count"]).reset_index()
            take=queries.loc[unresolved,cols].merge(agg,on=cols,how="left",sort=False)
        else:
            take=pd.DataFrame({"median":[history[target].median()]*unresolved.sum(),"count":[history[target].count()]*unresolved.sum()})
        ok=take["count"].fillna(0).to_numpy() >= q.PRIOR_MIN_SUPPORT; idx=np.flatnonzero(unresolved)
        values[idx[ok]]=take.loc[ok,"median"].to_numpy(float); supported[idx[ok]]=True
    if clip is not None: values=np.clip(values,*clip)
    return values,supported

def build_priors(history: pd.DataFrame, valid: pd.DataFrame) -> pd.DataFrame:
    """Strict month-forward priors; validation IDs must be absent from raw history."""
    if np.isin(valid.row_id.to_numpy(), history.row_id.to_numpy()).any(): raise RuntimeError("validation ID in history")
    h=history.loc[history.runway.notna()].copy(); v=valid.copy(); supported=v.runway.notna().to_numpy()
    offsets=q.historical_plan_offsets(h)
    b,ok=_resolve_vector(offsets,v,"plan_offset_seconds",[["airport","operator","flight","destination"],["airport","operator"],["airport"]],q.PLAN_OFFSET_CLIP)
    uevents=q.historical_unimpeded_events(h)
    u,uok=_resolve_vector(uevents,v,"unimpeded_seconds",[["airport","supplied_stand","runway","wake_type","quarter_hour_of_day"],["airport","runway","wake_type"],["airport","runway"],["airport"]],q.UNIMPEDED_CLIP)
    sevents=q.historical_service_events(h)
    s,sok=_resolve_vector(sevents,v,"service_interval_seconds",[["airport","runway"],["airport"]],q.SERVICE_CLIP)
    supported &= ok & uok & sok & v.plan_time.notna().to_numpy()
    out=pd.DataFrame({"row_id":v.row_id,"airport":v.airport,"runway":v.runway,"utc_day":v.plan_time.dt.floor("D"),
        "bhat":v.plan_time.astype("int64")/1e9+b,"unimpeded_seconds":u,"service_seconds":s,"supported":supported})
    out["deployable_arrival_time"]=out.bhat+out.unimpeded_seconds
    return out

def simulate_variable_service(frame: pd.DataFrame, arrivals: pd.Series | np.ndarray | None=None) -> pd.Series:
    w=frame.loc[frame.supported].copy()
    if w.runway.isna().any(): raise RuntimeError("missing runway entered queue")
    w["queue_arrival_time"] = w.deployable_arrival_time if arrivals is None else np.asarray(arrivals)[w.index]
    w=w.sort_values(["queue_arrival_time","row_id"],kind="mergesort"); free={}; pred={}
    for r in w.itertuples():
        key=(r.airport,r.runway); start=max(r.queue_arrival_time,free.get(key,-np.inf)); finish=start+r.service_seconds
        if finish-start != r.service_seconds: raise RuntimeError("service conservation failure")
        if key in free and start < free[key]: raise RuntimeError("service overlap")
        free[key]=finish; pred[r.Index]=r.unimpeded_seconds+(start-r.queue_arrival_time)
    return pd.Series(pred,dtype=float)

def _bootstrap(y, base, arm, days, seed=q.CONTROL_SEEDS["day_bootstrap"], draws=2000) -> tuple[float,float]:
    tab=pd.DataFrame({"y":y,"b":base,"a":arm,"day":days}).dropna(); groups=list(tab.groupby("day",sort=True))
    rng=np.random.default_rng(seed); gains=[]
    for _ in range(draws):
        picks=rng.integers(0,len(groups),len(groups)); s=pd.concat([groups[i][1] for i in picks],ignore_index=True)
        gains.append(math.sqrt(np.mean((s.y-s.b)**2))-math.sqrt(np.mean((s.y-s.a)**2)))
    return tuple(np.quantile(gains,[.025,.975]))

def _aggregate(arm_name, fold, valid, supported, y, base, pred) -> list[dict]:
    slices={"overall":np.ones(len(y),bool),"bulk":y<=3600,"tail":y>3600,
            "no_nm":valid.FLIGHT_ID_mvt.isna().to_numpy(),"supported":supported,"unsupported":~supported}
    rows=[]
    for name,mask in slices.items():
        for airport in [None]+sorted(valid.airport.dropna().unique()) if name=="overall" else [None]:
            use=mask if airport is None else mask & valid.airport.eq(airport).to_numpy()
            if not use.any(): continue
            rb=math.sqrt(np.mean((y[use]-base[use])**2)); ra=math.sqrt(np.mean((y[use]-pred[use])**2))
            lo,hi=_bootstrap(y[use],base[use],pred[use],valid.plan_time.dt.floor("D").to_numpy()[use])
            rows.append(dict(zip(q.OUTPUT_COLUMNS,[arm_name,fold,name,airport or "ALL",int(use.sum()),int((supported&use).sum()),rb,ra,rb-ra,lo,hi])))
    return rows

def _run_fold(fold: str) -> tuple[list[dict],dict]:
    from prc.data import month_file
    month=FOLDS[fold]; earlier=[f"2025-{i:02d}" for i in range(1,int(month[-2:]))]
    comp_path=ROOT/_load_json(BINDING)["folds"][fold]["path"]
    comp=pd.read_parquet(comp_path); base=comp.y_pred.to_numpy(float); ids=comp.MVT_ID_mvt.to_numpy()
    full=pd.read_parquet(month_file(month),columns=LABEL_COLUMNS); valid=prepare_raw(full,month)
    q.assert_exact_row_alignment(ids,valid.row_id.to_numpy())
    y=valid.TAXITIME_SEC_mvt.to_numpy(float)
    if not earlier: return _aggregate("deployable",fold,valid,np.zeros(len(valid),bool),y,base,base),{"fold":fold,"unsupported_all":True}
    hist=pd.concat([prepare_raw(pd.read_parquet(month_file(m),columns=LABEL_COLUMNS),m) for m in earlier],ignore_index=True)
    pri=build_priors(hist,valid); excluded=q.exclusion_mask(valid.row_id); pri.loc[excluded,"supported"]=False
    supported=pri.supported.to_numpy(); arms={}
    arms["deployable"]=simulate_variable_service(pri)
    oracle_arr=pri.deployable_arrival_time.copy()
    oracle_frame=pri.loc[supported].assign(true_block_time=valid.loc[supported,"block_time"].to_numpy())
    oracle_arr.loc[supported]=q.oracle_order_only_arrivals(oracle_frame)
    arms["oracle"]=simulate_variable_service(pri,oracle_arr)
    arms["shifted_2h"]=simulate_variable_service(pri,pri.deployable_arrival_time+7200)
    perm=q.permute_runways_within_airport_day(pri.loc[supported]); ppri=pri.copy(); ppri.loc[supported,"runway"]=perm
    arms["runway_permuted"]=simulate_variable_service(ppri)
    rows=[]; receipts={"fold":fold,"rows":len(valid),"supported":int(supported.sum()),"exclusion_matches":int(excluded.sum())}
    for name,series in arms.items():
        pred=base.copy(); pred[series.index.to_numpy()]=series.to_numpy(); rows += _aggregate(name,fold,valid,supported,y,base,pred)
    return rows,receipts

def run_phase_b() -> dict:
    pre=phase_b_preflight()  # must precede every label source read
    all_rows=[]; receipts=[]
    try:
        for fold in FOLDS:
            rows,receipt=_run_fold(fold); all_rows += rows; receipts.append(receipt)
        out=pd.DataFrame(all_rows,columns=q.OUTPUT_COLUMNS)
        primary=out[(out.arm=="deployable")&(out.slice=="overall")&(out.airport=="ALL")].set_index("fold")
        weights=_load_json(BINDING)["board_weights"]
        rb=math.sqrt(sum(weights[f]*primary.loc[f,"rmse_k2"]**2 for f in ("july_loo","january")))
        ra=math.sqrt(sum(weights[f]*primary.loc[f,"rmse_arm"]**2 for f in ("july_loo","january")))
        oracle=out[(out.arm=="oracle")&(out.slice=="overall")&(out.airport=="ALL")].set_index("fold")
        ora=math.sqrt(sum(weights[f]*oracle.loc[f,"rmse_arm"]**2 for f in ("july_loo","january")))
        status=q.endpoint_status(rb-ra,rb-ora,primary.loc["july_loo","ci95_low"],primary.loc["january","ci95_low"],primary.loc["december","gain_seconds"])
        EXP.joinpath("results").mkdir(exist_ok=True); out.to_csv(EXP/"results"/"aggregates.csv",index=False)
        receipt={"status":status,"primary_gain":rb-ra,"oracle_gain":rb-ora,"preflight":pre,"folds":receipts}
        (EXP/"results"/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
        return receipt
    except Exception as exc:
        EXP.joinpath("results").mkdir(exist_ok=True)
        (EXP/"results"/"INCOMPLETE.json").write_text(json.dumps({"incomplete":True,"error_type":type(exc).__name__})+"\n")
        raise

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("run",)); ap.parse_args()
    print(json.dumps(run_phase_b(),indent=2))

if __name__ == "__main__": main()
