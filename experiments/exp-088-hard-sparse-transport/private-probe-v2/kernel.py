import pathlib,subprocess,sys,tarfile,tempfile
p=next(pathlib.Path('/kaggle/input').glob('**/exp088-repair.bin'));d=tempfile.mkdtemp()
with tarfile.open(p,'r:gz') as t:t.extractall(d,filter='data')
subprocess.check_call([sys.executable,str(pathlib.Path(d)/'repo/experiments/exp-088-hard-sparse-transport/code/controller.py'),'remote-probe'])
