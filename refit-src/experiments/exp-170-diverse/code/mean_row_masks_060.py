"""Reuse the checked seed-mask mean for the checked row-control mean kind."""
import argparse
import json
from pathlib import Path

import average_seeds_n as average

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fit', type=Path, required=True)
    ap.add_argument('--parents', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    fit = json.loads((a.fit / 'FIT.json').read_text())
    assert fit['kind'] == 'equal n-seed row-control prediction mean'
    assert fit['args']['mode'] == 'row' and not fit['args']['final']
    average.KIND = fit['kind']
    a.out.mkdir(parents=True, exist_ok=False)
    average.masks(a)

if __name__ == '__main__':
    main()
