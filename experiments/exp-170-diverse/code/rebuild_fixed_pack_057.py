"""Rebuild the exact served thirty-parent held recipe on reviewed fixed Z inputs."""
import argparse
import concurrent.futures
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path
from check_time_day_052 import sha

FAST = "4e86dc93cf5a698b1bc87fc875ccb998a7243eb4c91df228551d1c51b6724714"
INPUT = "d052ad0239f6ebca4c05320eac7ba0766dbbfcd158afc06a318fa880ce92d3b1"
TRAIN = "547f8c8e482c2af473e2554c06b3aa88a78b643dcae3b863dff740129cccf87e"
HELPERS = {
"average_size_027.py": "082826acd35d2569c97fc552c56d98157e2fd76b4a14c2eee8617ab4b0edac1b",
"average_mixed_033.py": "05385518b5310bd9c724d192f44fd6402cfdb7573f448cf4d4d7843f7198d63a",
"predict_mask_size_027.py": "df07e600028053e3c551794f616d5c4246315d53ed4271b0cbd089d37f156c91",
"fit_runway_size_026.py": "d11f4f9cadd03c7b039f626fba5ba45ee6ccb288e371c4691952c3e3658f4c1d",
"join_masks.py": "bbe13f6209fc31aa4945a32eba844fe74fff54535553e2cd4e01e7058350ae3d",
"combine_masks_033.py": "7463b2fcfd0e7b522ee75dbd8608c836072705540d6025bbb58e4dc3c68ace99",
}


def verify_fit(folder, seed, width, epochs):
    for stage, weights in [("last", "last iterate"), ("ema2", "exponential mean, decay 0.9997"),
                           ("", "exponential mean, decay 0.999")]:
        path = folder / stage
        rec = json.loads((path / "FIT.json").read_text())
        assert rec["args"]["seed"] == seed and rec["args"]["width"] == width
        assert rec["args"]["epochs"] == epochs and rec["args"]["layers"] == 3
        assert rec["args"]["batch_size"] == 512 and rec["args"]["loss"] == "huber"
        assert rec["args"]["final"] is False and rec["weights"] == weights
        assert rec["inputs_manifest_sha256"] == INPUT and rec["code_sha256"] == FAST
        assert rec["train_ids_sha256"] == TRAIN and rec["train_rows"] == 1740628
        assert rec["train_months"] == [2,3,4,5,6,8,9,10,11,12]
        assert len(rec["history"]) == epochs and rec["history"][-1]["epoch"] == epochs
        for name, digest in rec["files"].items():
            assert sha(path / name) == digest
        assert rec["peak_gpu_bytes"] <= 10 * 1024 ** 3
    return {"path": str((folder / "last").resolve()), "FIT_sha256": sha(folder / "last/FIT.json")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--review-sha", required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--spec-sha", required=True)
    a = ap.parse_args()
    assert sha(a.review) == a.review_sha and "PASS" in a.review.read_text() and sha(__file__) in a.review.read_text()
    assert sha(a.spec) == a.spec_sha
    assert sha(Path("code/fit_runway_fast_029.py")) == FAST
    for name, digest in HELPERS.items():
        assert sha(Path("code") / name) == digest, name
    assert sha(Path("REVIEW-TIME-DAY-INPUTS-ACTUAL-052.md")) == "af9c2b25732f89ff8233c666a3f41b3d38c964abc0d4e11159513acbf3fd0da3"
    assert sha(Path("REVIEW-TIME-DAY-SELECTION-ACTUAL-052.md")) == "b8c9eaa96bacf853bbc4fbe912d0ad73bc1828c4bae5538e8c14491483230062"
    assert sha(Path("results/TIME-DAY-SELECTION-052.json")) == "833e5ff6e3fb1b8f435927508e3b5462f2e60bd3d6813d612671e84a429ecdb5"
    assert json.loads(Path("results/TIME-DAY-SELECTION-052.json").read_text())["chosen_form"] == "Z"
    inputs = Path("inputs/time-day-Z-052")
    assert sha(inputs / "MANIFEST.json") == INPUT
    manifest = json.loads((inputs / "MANIFEST.json").read_text())
    for name, digest in manifest["files"].items():
        assert sha(inputs / name) == digest
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        assert os.environ[name] == "2"
    assert os.getpriority(os.PRIO_PROCESS, 0) == 19 and set(os.sched_getaffinity(0)) == {14,15}
    pilot_path = Path("results/COHERENT-DARK-PILOT-053.json")
    pilot = json.loads(pilot_path.read_text())
    assert pilot["status"] == "PASS" and pilot["fixed_form"] == "Z" and not pilot["errors"]
    assert not Path("/proc", Path("logs/darkday-pilot-053.pid").read_text().strip()).exists()
    existing = pilot["streams"]["fixed6"]
    assert len(existing) == 6
    for seed, p in zip(range(17001,17007), existing):
        assert sha(Path(p["path"]) / "FIT.json") == p["FIT_sha256"]
        checked = verify_fit(Path(p["path"]).parent, seed, 128, 48)
        assert checked["path"] == p["path"]
        for airport in ["LIRF", "EDDF"]:
            rec = json.loads((Path(p["masks"][airport]) / "PREDICT.json").read_text())
            assert rec["fit_sha256"] == p["FIT_sha256"] and rec["unaffected_predictions_exact"] and rec["label_free"]
            assert sha(Path(p["masks"][airport]) / "jjrefit.parquet") == rec["files"]["jjrefit.parquet"]
    original = Path("out/sequence-close30-final-039/FIT.json")
    assert sha(original) == "6b21654958e980719b3db950deb2881508383b756e5d2a2c5fbe55d933da0a82"
    old = json.loads(original.read_text())
    assert old["n_fits"] == 30 and len(old["parents"]) == 30
    expected = {(seed,256,24) for seed in range(17001,17013)} | {(seed,128,48) for seed in range(17001,17013)} | {(seed,256,48) for seed in range(17001,17007)}
    assert {(p["args"]["seed"],p["args"]["width"],p["args"]["epochs"]) for p in old["parents"]} == expected
    assert all(p["args"]["layers"] == 3 and p["args"]["batch_size"] == 512 and p["code_sha256"] == FAST and p["weights_form"] == "last iterate" for p in old["parents"])
    for recipe, seeds in [("wide24",range(17001,17013)),("r48",range(17007,17013)),("wide48",range(17001,17007))]:
        assert all(not Path(f"out/sequence-fixed-{recipe}-{seed}-057").exists() for seed in seeds)
    Path("logs/fixed-pack-057.claim").mkdir(exist_ok=False)
    Path("logs/fixed-pack-057.pid").write_text(str(os.getpid()) + "\n")
    record = {"status":"RUNNING","source_row":902,"source_sha256":sha(__file__),"spec_sha256":sha(a.spec),"review_sha256":sha(a.review),"pilot_sha256":sha(pilot_path),"input_manifest_sha256":INPUT,"original039_FIT_sha256":sha(original),"existing_fixed6":existing,"stage":"LAST only","quality_read":False,"pack_changed":False,"annual_fits":False,"streams":{}}
    def run(argv, logname, gpu):
        used = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                          "--format=csv,noheader,nounits"], text=True).strip())
        assert used <= 14000, f"GPU{gpu} has {used} MiB used"
        with Path("logs", logname).open("x") as log:
            process = subprocess.Popen([sys.executable, "-u", *argv],
                                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                       stdout=log, stderr=subprocess.STDOUT)
            print("PROCESS", gpu, process.pid, argv, flush=True)
            assert process.wait() == 0, f"failed: {argv}"

    def masks(parent, inputs, prefix, gpu):
        result = {}
        for airport in ["LIRF", "EDDF"]:
            out = Path("out", prefix + "-mask-" + airport.lower())
            run(["code/predict_mask_size_027.py", "--inputs", str(inputs), "--fit", str(parent),
                 "--airport", airport, "--out", str(out)], out.name + ".log", gpu)
            rec = json.loads((out / "PREDICT.json").read_text())
            assert rec["fit_sha256"] == sha(parent / "FIT.json")
            assert rec["label_free"] and rec["unaffected_predictions_exact"]
            assert rec["peak_gpu_bytes"] <= 10 * 1024 ** 3
            assert sha(out / "jjrefit.parquet") == rec["files"]["jjrefit.parquet"]
            result[airport] = str(out.resolve())
        joint = Path("out", prefix + "-mask-joint")
        run(["code/join_masks.py", "--fit", str(parent), "--lirf", result["LIRF"],
             "--eddf", result["EDDF"], "--out", str(joint)], joint.name + ".log", gpu)
        receipt = json.loads((joint / "PREDICT.json").read_text())
        assert receipt["fit_sha256"] == sha(parent / "FIT.json")
        assert receipt["label_free"] and receipt["unaffected_predictions_exact"]
        assert sha(joint / "jjrefit.parquet") == receipt["files"]["jjrefit.parquet"]
        result["joint"] = str(joint.resolve())
        result["joint_PREDICT_sha256"] = sha(joint / "PREDICT.json")
        result["joint_vector_sha256"] = sha(joint / "jjrefit.parquet")
        return result

    def fit(seed, recipe, width, epochs, gpu):
        folder = Path(f"out/sequence-fixed-{recipe}-{seed}-057")
        run(["code/fit_runway_fast_029.py", "--inputs", str(inputs), "--out", str(folder),
             "--mode", "sequence-runway", "--loss", "huber", "--epochs", str(epochs), "--batch-size", "512",
             "--seed", str(seed), "--width", str(width), "--layers", "3", "--ema", ".999", "--ema2", ".9997"], folder.name + ".log", gpu)
        p = verify_fit(folder, seed, width, epochs)
        p["masks"] = masks(Path(p["path"]), inputs, folder.name, gpu)
        return p

    def stream0():
        return [fit(seed,"wide24",256,24,0) for seed in range(17001,17013)]

    def stream1():
        return {"r48": [fit(seed,"r48",128,48,1) for seed in range(17007,17013)],
                "wide48": [fit(seed,"wide48",256,48,1) for seed in range(17001,17007)]}

    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {"wide24":pool.submit(stream0),"r48_wide48":pool.submit(stream1)}
        for key,job in jobs.items():
            try: record["streams"][key] = job.result()
            except Exception as exc: errors.append(f"{key}: {type(exc).__name__}: {exc}")
    if not errors:
        groups = {"wide24":record["streams"]["wide24"],"r48":existing+record["streams"]["r48_wide48"]["r48"],"wide48":record["streams"]["r48_wide48"]["wide48"]}
        record["means"] = {}
        for name,parents in groups.items():
            mean = Path(f"out/fixed-{name}-bag{len(parents)}-057")
            run(["code/average_size_027.py","--kind","fits","--parents",*[p["path"] for p in parents],"--out",str(mean)],mean.name+".log",0)
            masked = {}
            for airport in ["LIRF","EDDF"]:
                out = Path(str(mean)+"-mask-"+airport.lower())
                run(["code/average_size_027.py","--kind","masks","--fit",str(mean),"--parents",*[p["masks"][airport] for p in parents],"--out",str(out)],out.name+".log",0)
                masked[airport] = str(out)
            joint = Path(str(mean)+"-mask-joint")
            run(["code/join_masks.py","--fit",str(mean),"--lirf",masked["LIRF"],"--eddf",masked["EDDF"],"--out",str(joint)],joint.name+".log",0)
            record["means"][name] = {"path":str(mean.resolve()),"FIT_sha256":sha(mean/"FIT.json"),"masks":masked,"joint_path":str(joint.resolve())}
        mixed = Path("out/fixed-close30-057")
        parents = groups["wide24"] + groups["r48"] + groups["wide48"]
        assert len(parents) == 30 and len({p["path"] for p in parents}) == 30
        run(["code/average_mixed_033.py","--parents",*[p["path"] for p in parents],"--out",str(mixed)],mixed.name+".log",0)
        record["mixed"] = {"path":str(mixed.resolve()),"FIT_sha256":sha(mixed/"FIT.json"),"masks":{}}
        for view in ["LIRF","EDDF","joint"]:
            out = Path(str(mixed)+"-mask-"+view.lower())
            plain = [f"{record['means'][name]['path']}:{len(groups[name])}" for name in groups]
            masked = [f"{record['means'][name]['joint_path'] if view=='joint' else record['means'][name]['masks'][view]}:{len(groups[name])}" for name in groups]
            run(["code/combine_masks_033.py","--fit",str(mixed),"--masked",*masked,"--unmasked",*plain,"--out",str(out)],out.name+".log",0)
            record["mixed"]["masks"][view] = {"path":str(out.resolve()),"PREDICT_sha256":sha(out/"PREDICT.json")}
    record["errors"] = errors
    record["status"] = "PASS" if not errors else "FAIL"
    record["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    Path("results/FIXED-PACK-REBUILD-057.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps({"status":record["status"],"errors":errors}),flush=True)
    assert not errors,errors


if __name__ == "__main__": main()
