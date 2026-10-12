from pathlib import Path
import importlib.util
p=list(Path('/kaggle/input').glob('**/exp089_controller.py'))
if len(p)!=1:raise RuntimeError('one controller required')
s=importlib.util.spec_from_file_location('exp089_controller',p[0]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);m.remote_probe()
