"""exp-186: mode caller of the reviewed check_annual.py (independent check of the g annual inputs).

check_annual.py is pinned by SHA-256 (annual_job.py, REVIEW-ANNUAL-SOURCE.json) and stays unchanged. This file
reads the reviewed source, checks its SHA-256, sets the one ranking-count literal (`len(ids)!=344841`) to
prc.data.n_ranking() (prepare_annual_186.reviewed) and runs its main() with the same arguments. In the ranking
mode the code that runs is byte-equal to the reviewed file. The result keeps the field `checker_sha256` of the
reviewed file; CHECK-CALLER-186.json next to the result records this caller and the count.

    PRC_RANKING_MODE=final python code/check_annual_186.py --inputs <annual inputs> --out <checks.json>
"""
import argparse
import json
import sys
from pathlib import Path

from prepare_annual_186 import D, reviewed, sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    side = a.out.with_name(a.out.stem + '-CALLER-186.json')
    if a.out.exists() or side.exists():
        raise ValueError('Annual check exists')
    mode = D.ranking_mode()
    module, record = reviewed('check_annual.py')
    sys.argv = [record['source'], '--inputs', str(a.inputs), '--out', str(a.out)]
    module['main']()
    result = json.loads(a.out.read_text())
    if result['status'] != 'PASS' or result['ranking_rows'] != D.n_ranking():
        raise ValueError('annual check did not pass for the mode')
    side.write_text(json.dumps({'what': 'g annual check of the mode through the reviewed check_annual.py (exp-186)',
                                'ranking_mode': mode, 'reviewed_source': record, 'checks': str(a.out),
                                'checks_sha256': sha(a.out), 'caller_sha256': sha(__file__)}, indent=2) + '\n')


if __name__ == '__main__':
    main()
