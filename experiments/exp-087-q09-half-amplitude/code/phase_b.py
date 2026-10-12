"""Complete disabled Phase-B workflow for exp-087."""
from __future__ import annotations
import argparse, hashlib, importlib.util, io, json, math, os, re, shutil, subprocess, sys, tarfile, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent; EXP=HERE.parent; ROOT=EXP.parents[1]
sp=importlib.util.spec_from_file_location("half087",HERE/"half_amplitude.py"); h=importlib.util.module_from_spec(sp); sp.loader.exec_module(h)
MANIFEST=EXP/"source-manifest.json"; ACCEPTANCE=EXP/"review-acceptance.json"; STATE=EXP/"measurement-state.json"; INCOMPLETE=EXP/"INCOMPLETE.json"; RESULT=EXP/"analysis/result.json"; LANDED=EXP/"landed/july"
REGISTRATION_COMMIT="040b103b24db720389e0111b7c9eec55399547cd"; REVIEW_SHA="f94f61f40ff37d66cddac927517d69f991d62c887198b3a5d1fb45f7e3231e61"
ACCEPTANCE_KEYS={"experiment","registration_commit","review_file","review_sha256","review_verdict","accepted_findings","repair_commit","source_manifest_sha256","authority"}
STATES=("unused","july_package_prepared","july_launch_started","july_landed","july_verified","score_started","complete")
def read_json(path): return json.loads(Path(path).read_text())
def state(): return read_json(STATE) if STATE.exists() else {"state":"unused"}
def transition(prior,new,extra=None):
    if state().get("state")!=prior or new not in STATES: raise RuntimeError(f"state must be {prior}")
    h.atomic_json(STATE,{"state":new,**(extra or {})})
def record_failure(exc):
    if not INCOMPLETE.exists(): h.atomic_json(INCOMPLETE,{"incomplete":True,"error_type":type(exc).__name__},exclusive=True)
def run_guarded(prior,boundary,action):
    if INCOMPLETE.exists() or state()["state"]=="complete": raise RuntimeError("measurement is terminal")
    transition(prior,boundary)
    try: return action()
    except BaseException as exc: record_failure(exc); raise
def require_acceptance():
    if not ACCEPTANCE.is_file(): raise PermissionError("Phase B has no desk acceptance")
    a=read_json(ACCEPTANCE)
    if set(a)!=ACCEPTANCE_KEYS: raise PermissionError("acceptance schema mismatch")
    exact={"experiment":"exp-087","registration_commit":REGISTRATION_COMMIT,"review_file":"review/round1.md","review_sha256":REVIEW_SHA,"review_verdict":"# BLOCK","accepted_findings":list(range(1,11)),"authority":"AUTHORIZE PHASE B"}
    if any(a.get(k)!=v for k,v in exact.items()): raise PermissionError("acceptance binding mismatch")
    if not re.fullmatch(r"[0-9a-f]{40}",a["repair_commit"]) or not re.fullmatch(r"[0-9a-f]{64}",a["source_manifest_sha256"]): raise PermissionError("bad acceptance digest")
    if h.sha256_file(EXP/a["review_file"])!=REVIEW_SHA or h.sha256_file(MANIFEST)!=a["source_manifest_sha256"]: raise PermissionError("registered hash mismatch")
    tracked=subprocess.run(["git","show",f"HEAD:{ACCEPTANCE.relative_to(ROOT)}"],cwd=ROOT,capture_output=True,check=True).stdout
    if tracked!=ACCEPTANCE.read_bytes(): raise PermissionError("acceptance is not exact HEAD bytes")
    for commit in (REGISTRATION_COMMIT,a["repair_commit"]): subprocess.run(["git","merge-base","--is-ancestor",commit,"HEAD"],cwd=ROOT,check=True)
    return a
def native_id_record(path):
    ids=pd.read_parquet(path,columns=["MVT_ID_mvt"]).MVT_ID_mvt.to_numpy()
    if len(ids)==0 or not np.issubdtype(ids.dtype,np.number) or not np.isfinite(ids).all() or len(np.unique(ids))!=len(ids): raise ValueError("invalid native IDs")
    return {"rows":len(ids),"id_dtype":str(ids.dtype),"ordered_id_sha256":hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()},ids
def preflight():
    require_acceptance(); m=read_json(MANIFEST)
    for rel,rec in m["sources"].items():
        path=ROOT/rel
        if h.sha256_file(path)!=rec["sha256"]: raise RuntimeError(f"source mutation: {rel}")
        if "rows" in rec and native_id_record(path)[0]!={k:rec[k] for k in ("rows","id_dtype","ordered_id_sha256")}: raise RuntimeError(f"source identity mismatch: {rel}")
    missing=set(m["runtime_source_trace"])-set(m["sources"])
    if missing: raise RuntimeError(f"source trace omitted {sorted(missing)}")
    return m
def deterministic_package(source_commit,out):
    with tempfile.TemporaryDirectory() as td:
        repo=Path(td)/"repo"; repo.mkdir(); archive=subprocess.run(["git","archive",source_commit],cwd=ROOT,capture_output=True,check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(archive)) as tf: tf.extractall(repo)
        (repo/"GIT_COMMIT.txt").write_text(f"{source_commit}\nclean-tree=true\n")
        with tarfile.open(out,"w") as tf:
            for path in sorted(repo.rglob('*')):
                info=tf.gettarinfo(str(path),str(Path('repo')/path.relative_to(repo))); info.uid=info.gid=0; info.uname=info.gname=''; info.mtime=0
                if path.is_file():
                    with path.open('rb') as src: tf.addfile(info,src)
                else: tf.addfile(info)
    return h.sha256_file(out)
def prepare_package():
    m=preflight(); out=EXP/"package/prc-repo.tar.gz"; out.parent.mkdir(parents=True,exist_ok=True)
    digest=deterministic_package(m["july_reproduction"]["package_source_commit"],out)
    if digest!=m["july_reproduction"]["package_sha256"]: raise RuntimeError("package digest mismatch")
    transition("unused","july_package_prepared",{"package_sha256":digest})
def launch_july():
    m=preflight()
    def action():
        if float(os.getloadavg()[0])>10: raise RuntimeError("one-minute Mac load exceeds 10")
        if h.sha256_file(EXP/"package/prc-repo.tar.gz")!=m["july_reproduction"]["package_sha256"]: raise RuntimeError("prepared package changed")
        subprocess.run(m["july_reproduction"]["publish_command"],cwd=ROOT,check=True); subprocess.run(m["july_reproduction"]["verify_dataset_command"],cwd=ROOT,check=True); subprocess.run(m["july_reproduction"]["kernel_command"],cwd=ROOT,check=True)
    run_guarded("july_package_prepared","july_launch_started",action)
def mark_landed():
    preflight()
    if not LANDED.resolve().is_relative_to(EXP.resolve()): raise RuntimeError("landed path escapes experiment")
    transition("july_launch_started","july_landed")
RECEIPT_KEYS={"kernel_slug","kernel_version","status","wall_seconds","image","cpu_shape","private","offline","accelerator","code_dataset_version","code_dataset_sha256","data_dataset_version","data_dataset_sha256","package_sha256","outputs"}
OUTPUT_KEYS={"path","sha256","rows","schema","dtypes","id_dtype","ordered_id_sha256","rmse"}
def verify_july_files():
    m=preflight(); raw=read_json(LANDED/"receipt.json"); expected=m["july_reproduction"]
    if set(raw)!=RECEIPT_KEYS or raw["status"]!="COMPLETE" or not(raw["private"] and raw["offline"] and raw["accelerator"]=="CPU"): raise RuntimeError("strict July receipt rejected")
    if isinstance(raw["wall_seconds"],bool) or not isinstance(raw["wall_seconds"],(int,float)) or not math.isfinite(raw["wall_seconds"]) or raw["wall_seconds"]<=0: raise RuntimeError("invalid wall time")
    for key in ("kernel_slug","kernel_version","image","cpu_shape","code_dataset_version","code_dataset_sha256","data_dataset_version","data_dataset_sha256","package_sha256"):
        if raw[key]!=expected[key]: raise RuntimeError(f"July metadata mismatch: {key}")
    if set(raw["outputs"])!={"none","chain","derived"} or not math.isfinite(expected["rmse_tolerance"]): raise RuntimeError("July arm or tolerance mismatch")
    clean={k:raw[k] for k in RECEIPT_KEYS-{"outputs"}}; clean["outputs"]={}
    for arm,rec in raw["outputs"].items():
        if set(rec)!=OUTPUT_KEYS: raise RuntimeError("strict output receipt rejected")
        path=(LANDED/rec["path"]).resolve()
        if not path.is_relative_to(LANDED.resolve()) or not path.is_file() or h.sha256_file(path)!=rec["sha256"]: raise RuntimeError("output path or digest mismatch")
        frame=pd.read_parquet(path); got,_=native_id_record(path)
        if (list(frame.columns)!=rec["schema"] or rec["schema"]!=expected["schema"] or
            {c:str(frame[c].dtype) for c in frame}!=rec["dtypes"] or rec["dtypes"]!=expected["dtypes"] or
            rec["sha256"]!=expected["expected_output_sha256"][arm] or
            got!={"rows":rec["rows"],"id_dtype":rec["id_dtype"],"ordered_id_sha256":rec["ordered_id_sha256"]} or got!=expected["input_identity"]): raise RuntimeError("output schema or identity mismatch")
        y,pred=h.validate_numeric(frame.y_true.to_numpy(),frame.y_pred.to_numpy()); rmse=float(np.sqrt(np.mean((y-pred)**2)))
        if not math.isfinite(rec["rmse"]) or abs(rmse-rec["rmse"])>expected["rmse_tolerance"] or abs(rmse-expected["reference_rmse"][arm])>expected["rmse_tolerance"]: raise RuntimeError("recomputed RMSE mismatch")
        clean["outputs"][arm]={"sha256":rec["sha256"],**got,"rmse":rmse}
    return clean
def verify_july():
    clean=run_guarded("july_landed","july_verified",verify_july_files); h.atomic_json(EXP/"receipts/july-sanitized.json",clean)
def reconstruct_served_chain(july_path):
    """Use exp-040's exact frames_for -> anchor -> stacker -> holds -> masks path."""
    sys.path[:0]=[str(ROOT/"experiments/exp-040-chain/code"),str(ROOT)]; import q09 as Q
    with tempfile.TemporaryDirectory() as td:
        stage=Path(td); shutil.copyfile(july_path,stage/"chain--july_loo.parquet")
        for fold in ("december","january"): os.symlink(ROOT/f"experiments/exp-040-chain/oof/chain--{fold}.parquet",stage/f"chain--{fold}.parquet")
        old=Q.ARM_OOF; Q.ARM_OOF=str(stage/"{arm}--{fold}.parquet")
        try:
            ref=Q.Sc.reference(); apt={f:ref[f].airport.astype(str).to_numpy() for f in Q.FOLDS}; v7=Q.C.eham_mask({f:ref[f].y_pred_v6.to_numpy("float64") for f in Q.FOLDS},ref)[0]
            data0=Q.frames_for(None); a5=Q.C.anchors(data0,Q.NAMES5); blk={f:Q.SW.block_for(f,Q.SW.panel_of(data0[f]),{"blocks":[Q.C.BLOCK],"win":Q.C.WIN}) for f in Q.FOLDS}
            p10=Q.serve(data0,blk,ref,v7,apt); chain=Q.serve(Q.frames_for("chain"),blk,ref,v7,apt); noac={f:ref[f].regime.to_numpy()=="noac" for f in Q.FOLDS}
            return ref,p10,{f:np.where(noac[f],p10[f],chain[f]) for f in Q.FOLDS}
        finally: Q.ARM_OOF=old
def score_action():
    clean=read_json(EXP/"receipts/july-sanitized.json"); ref,p10,served=reconstruct_served_chain(LANDED/"chain--july_loo.parquet")
    sys.path[:0]=[str(ROOT/"experiments/exp-047-queue/code"),str(ROOT)]; import base as B, trim12h
    panel=B.load(with_v12=True); inc=np.load(ROOT/"experiments/exp-046-surfacefit/work/d-bag10-255-free.npz"); stats={}; parts={}; comp={}; trim={}
    for fold in h.FOLDS:
        ids=ref[fold].MVT_ID_mvt.to_numpy(); current=panel[fold].set_index("MVT_ID_mvt").reindex(ids); comparator=current.p12.to_numpy()+inc[fold]; full=served[fold]-p10[fold]; arm=h.half_update(comparator,full); y=ref[fold].y_true.to_numpy(); day=np.floor(ref[fold].t.to_numpy()/86400).astype('int64')
        stats[fold]=h.fold_stats(y,comparator,arm,day); stats[fold]["fold"]=fold; parts[fold]=(y,comparator,arm,day)
        month={"july_loo":"2025-07","december":"2025-12","january":"2025-01"}[fold]; block=pd.read_parquet(ROOT/f"artifacts/exp-047/block/{month}.parquet",columns=["MVT_ID_mvt","obs_has_clock"]); clock=pd.Series(block.obs_has_clock.fillna(False).to_numpy(bool),index=block.MVT_ID_mvt.to_numpy()).reindex(ids).fillna(False).to_numpy(bool); keep=~(clock|(inc[fold]!=0)); comp[fold]=h.fold_stats(y[keep],comparator[keep],arm[keep],day[keep],"complement"); keep=trim12h.keep(fold,ids); trim[fold]=h.fold_stats(y[keep],comparator[keep],arm[keep],day[keep],"trimmed_12h")
    weighted=h.board_record(stats); weighted["ci95_low"],weighted["ci95_high"]=h.board_bootstrap(parts); verdict=h.status(weighted["gain"],{f:stats[f]["gain"] for f in h.FOLDS},{f:stats[f]["ci95_low"] for f in h.FOLDS})
    result={"schema_version":2,"status":verdict,"primary":stats,"weighted_primary":weighted,"historical_full_amplitude":read_json(MANIFEST)["historical_full_amplitude"],"secondary":{"complement":comp,"trimmed_12h":trim},"july_provenance":clean,"cause":"fixed half-amplitude stand-chain update on the frozen current head","destination_value":"board-share RMSE gain with all three raw fold gates","path_feasibility":"fold evidence only; no candidate or upload is authorized","revival_condition":"a later registration only if this fixed form does not advance","claim_boundary":"exp-087 fold evidence only; no label-selected amplitude, masks, head or population","source_manifest_sha256":h.sha256_file(MANIFEST)}
    h.atomic_json(RESULT,result,exclusive=True); transition("score_started","complete",{"result_sha256":h.sha256_file(RESULT)}); return result
def score(): return run_guarded("july_verified","score_started",score_action)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=("prepare-package","launch-july","mark-landed","verify-july","score")); command=ap.parse_args().command
    {"prepare-package":prepare_package,"launch-july":launch_july,"mark-landed":mark_landed,"verify-july":verify_july,"score":score}[command]()
if __name__=="__main__": main()
