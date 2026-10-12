"""Render every registered exp-087 aggregate after verified completion."""
import os,tempfile
from pathlib import Path
from phase_b import RESULT,EXP,h,require_acceptance,state,read_json
FIELDS=("rows","changed_rows","rms_movement","max_movement","rmse_comparator","rmse_arm","gain","A","B","A_over_B","ci95_low","ci95_high")
def table(title,records):
    out=[f"## {title}","","| fold | rows | changed | RMS move | max move | comparator | arm | gain | A | B | A/B | 95% interval |","|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for fold,r in records.items():
        missing=set(FIELDS)-set(r)
        if missing: raise RuntimeError(f"result misses {sorted(missing)}")
        out.append(f"| {fold} | {r['rows']} | {r['changed_rows']} | {r['rms_movement']:.5f} | {r['max_movement']:.5f} | {r['rmse_comparator']:.5f} | {r['rmse_arm']:.5f} | {r['gain']:+.5f} | {r['A']:+.5f} | {r['B']:.5f} | {r['A_over_B']:.5f} | [{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}] |")
    return out
def main():
    require_acceptance(); s=state()
    if s.get('state')!='complete' or not RESULT.is_file() or h.sha256_file(RESULT)!=s.get('result_sha256'): raise RuntimeError('unverified result state')
    r=read_json(RESULT); required={"status","primary","weighted_primary","historical_full_amplitude","secondary","july_provenance","cause","destination_value","path_feasibility","revival_condition","claim_boundary","source_manifest_sha256"}
    if not required.issubset(r): raise RuntimeError('incomplete result')
    lines=["# exp-087 report","",f"Status: `{r['status']}`.",""]+table("Primary all-row result",r['primary'])
    lines += ["","## Weighted primary","",f"`{r['weighted_primary']}`","","## Historical full amplitude","",f"`{r['historical_full_amplitude']}`",""]
    lines += table("Report-only complement",r['secondary']['complement'])+[""]+table("Report-only trimmed-12h",r['secondary']['trimmed_12h'])
    lines += ["","## Sanitized July provenance","",f"`{r['july_provenance']}`","",f"Cause: {r['cause']}","",f"Destination value: {r['destination_value']}","",f"Path feasibility: {r['path_feasibility']}","",f"Revival condition: {r['revival_condition']}","",f"Claim boundary: {r['claim_boundary']}","",f"Source manifest SHA-256: `{r['source_manifest_sha256']}`.",f"Result SHA-256: `{s['result_sha256']}`.",""]
    fd,tmp=tempfile.mkstemp(prefix='.REPORT.',dir=EXP)
    try:
        with os.fdopen(fd,'w') as stream: stream.write('\n'.join(lines)); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,EXP/'REPORT.md')
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
if __name__=='__main__': main()
