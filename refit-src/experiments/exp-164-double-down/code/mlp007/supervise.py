"""Run one numbered phase; the Mac monitor has the verified wake callback."""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from io_utils import W, write, lane

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',choices=['prepare','full','physical','screen-full','screen-physical'],required=True);ap.add_argument('--attempt',type=int,default=1);a=ap.parse_args()
    name=a.job if a.attempt==1 else f'{a.job}-{a.attempt:02d}'
    lane();p=W/'results/mlp-007'/f'{name}-status.json'
    state={'state':'running','job':a.job,'attempt':a.attempt,'pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write(p,state)
    try:
        script='prepare.py' if a.job=='prepare' else ('wait_and_screen.py' if a.job.startswith('screen-') else 'fit.py')
        cmd=[sys.executable,str(Path(__file__).with_name(script))]
        if a.job!='prepare':cmd+=['--arm',a.job.removeprefix('screen-')]
        with (W/f'logs/mlp007-{name}.log').open('x') as f:
            child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT)
            state['child_pid']=child.pid;write(p,state,overwrite=True)
            rc=child.wait()
            if rc:raise RuntimeError(f'{a.job}: exit {rc}')
        state['state']='complete'
    except Exception as e:
        state.update(state='failed',error=str(e));raise
    finally:
        state['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());write(p,state,overwrite=True)

if __name__=='__main__':main()
