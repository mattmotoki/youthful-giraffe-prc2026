import hashlib,importlib.util,json,pathlib,shutil,tarfile,tempfile
PACKAGE_NAME='exp089-repair.tar.gz';EXPECTED_SHA256='aef08f1c0aa1d69d002583c53d96093967202d5d6c91af1770c12899ec965dc7';MAX_BYTES=500000000
root=pathlib.Path('/kaggle/input');matches=list(root.glob('*/'+PACKAGE_NAME))
if len(matches)!=1:raise RuntimeError('exact package filename')
package=matches[0]
if package.stat().st_size>MAX_BYTES or hashlib.sha256(package.read_bytes()).hexdigest()!=EXPECTED_SHA256:raise RuntimeError('outer package identity')
with tarfile.open(package,'r:gz') as archive:
 members=archive.getmembers();names=[m.name for m in members]
 if len(names)!=len(set(names)) or any(not m.isfile() or m.islnk() or m.issym() or '..' in pathlib.PurePosixPath(m.name).parts or m.size>MAX_BYTES for m in members):raise RuntimeError('unsafe member')
 manifest=json.loads(archive.extractfile('OUTER-MANIFEST.json').read())
 if set(manifest['files'])!=set(names)-{'OUTER-MANIFEST.json'}:raise RuntimeError('unexpected member')
 for name,digest in manifest['files'].items():
  if hashlib.sha256(archive.extractfile(name).read()).hexdigest()!=digest:raise RuntimeError('member mutation')
 target=pathlib.Path(tempfile.mkdtemp());archive.extractall(target,filter='data')
try:
 module=target/'experiments/exp-089-movement-set/repair/controller.py';spec=importlib.util.spec_from_file_location('isolated_exp089',module);controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller);controller.main(['real'])
finally:shutil.rmtree(target)
