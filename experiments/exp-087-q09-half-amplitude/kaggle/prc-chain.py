"""Frozen exp-087 July rerun; byte-identical producer body is imported from exp-040."""
from pathlib import Path
import shutil
import tarfile

repo=Path('/tmp/repo')
if repo.exists(): shutil.rmtree(repo)
tars=sorted(Path('/kaggle/input').rglob('prc-repo.tar.gz'))
if not tars: raise SystemExit('registered code package is absent')
with tarfile.open(tars[0]) as archive: archive.extractall('/tmp')
source = repo / "experiments/exp-040-chain/kernels/chain-july_loo/prc-chain.py"
if not source.is_file():
    raise SystemExit("committed July producer is absent from the code package")
code = compile(source.read_bytes(), str(source), "exec")
exec(code, {"__name__": "__main__", "__file__": str(source)})
