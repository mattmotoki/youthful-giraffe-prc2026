"""exp-124 OPDI probe: does OPDI's EVENT table give off-block-like events at the ten PRC airports?
Reads one 10-day July 2026 events file and the July 2026 flight list. Label-free: no PRC data is read."""
import time, urllib.request, hashlib, json
import pandas as pd, pyarrow.parquet as pq

BASE = "https://www.eurocontrol.int/performance/data/download/OPDI/v002"
FILES = {"ev": f"{BASE}/flight_events/flight_events_20260709_20260719.parquet",
         "fl": f"{BASE}/flight_list/flight_list_202607.parquet"}
APTS = ["EDDF", "EDDM", "EGLL", "EHAM", "LEBL", "LEMD", "LFPG", "LIRF", "LTFM", "LSZH"]
out = {}
for k, u in FILES.items():
    t = time.time(); p = f"/kaggle/working/{k}.parquet"
    urllib.request.urlretrieve(u, p)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    out[k] = {"url": u, "sha256": h, "seconds": round(time.time() - t, 1)}
    pf = pq.ParquetFile(p)
    print(k, "rows", pf.metadata.num_rows, "download s", out[k]["seconds"], "sha", h[:12])
    print(pf.schema_arrow)

ev = pd.read_parquet("/kaggle/working/ev.parquet")
fl = pd.read_parquet("/kaggle/working/fl.parquet")
print(ev.head(3).T); print(fl.head(3).T)
print(ev["type"].value_counts())
lc = {c.lower(): c for c in fl.columns}
adep = lc.get("adep"); fid = lc.get("id")
evid = next(c for c in ev.columns if c.lower() in ("flight_id", "flt_id", "id_flight"))
tcol = next(c for c in ev.columns if "time" in c.lower())
dep = fl[fl[adep].isin(APTS)][[fid, adep]].rename(columns={fid: "fid", adep: "adep"})
print("July flight-list departures at the ten airports:", dep.groupby("adep").size().to_dict())
e = ev[ev[evid].isin(dep["fid"])]
e = e.merge(dep, left_on=evid, right_on="fid")
days = pd.to_datetime(e[tcol]).dt.date
e = e[(days >= pd.Timestamp("2026-07-09").date()) & (days < pd.Timestamp("2026-07-19").date())]
rows = []
for a, g in e.groupby("adep"):
    n_flights = g["fid"].nunique()
    r = {"adep": a, "flights_with_any_event": n_flights}
    for typ in ["exit-parking_position", "entry-taxiway", "exit-taxiway", "entry-runway", "take-off"]:
        r[typ] = int(g.loc[g["type"] == typ, "fid"].nunique())
    rows.append(r)
tab = pd.DataFrame(rows).sort_values("adep")
print(tab.to_string(index=False))
# departure-side gap: take-off minus exit-parking_position, per airport
piv = e[e["type"].isin(["exit-parking_position", "take-off"])].copy()
piv["t"] = pd.to_datetime(piv[tcol])
first_exit = piv[piv["type"] == "exit-parking_position"].groupby("fid")["t"].min()
first_to = piv[piv["type"] == "take-off"].groupby("fid")["t"].min()
gap = (first_to - first_exit).dt.total_seconds().dropna()
gap = gap[(gap > 0) & (gap < 4 * 3600)]
g2 = gap.to_frame("gap").join(dep.set_index("fid")["adep"])
print(g2.groupby("adep")["gap"].describe(percentiles=[.1, .5, .9]).round(0).to_string())
tab.to_csv("/kaggle/working/coverage_2026-07-09_19.csv", index=False)
json.dump(out, open("/kaggle/working/sources.json", "w"), indent=1)
