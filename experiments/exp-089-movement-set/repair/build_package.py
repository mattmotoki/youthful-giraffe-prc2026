#!/usr/bin/env python3
"""Deterministic package builder. The public CLI accepts no identity or path."""
import gzip,hashlib,io,json,pathlib,subprocess,tarfile
ROOT=pathlib.Path(__file__).resolve().parents[3];EXP=ROOT/'experiments/exp-089-movement-set'
SOURCES=('experiments/exp-089-movement-set/repair/controller.py','experiments/exp-089-movement-set/repair/synthetic-input.json','experiments/exp-089-movement-set/phase-a-spec-v2.json','experiments/exp-089-movement-set/result-schema.json')
def blob(v):return (json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
def assemble(source_members,commit):
 members=dict(source_members);manifest={'schema':1,'source_commit':commit,'files':{p:hashlib.sha256(v).hexdigest() for p,v in members.items()}};members['OUTER-MANIFEST.json']=blob(manifest);raw=io.BytesIO()
 with tarfile.open(fileobj=raw,mode='w',format=tarfile.PAX_FORMAT) as t:
  for n,v in sorted(members.items()):i=tarfile.TarInfo(n);i.size=len(v);i.mode=0o644;i.mtime=i.uid=i.gid=0;i.uname=i.gname='';t.addfile(i,io.BytesIO(v))
 out=io.BytesIO()
 with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0,compresslevel=9) as z:z.write(raw.getvalue())
 return out.getvalue(),manifest
def build(commit):
 return assemble({p:subprocess.check_output(['git','show',f'{commit}:{p}'],cwd=ROOT) for p in SOURCES},commit)
def main():
 binding=json.loads((EXP/'binding-v2.json').read_text());raw,_=build(binding['package_source_commit'])
 if len(raw)!=binding['package']['bytes'] or hashlib.sha256(raw).hexdigest()!=binding['package']['sha256']:raise RuntimeError('bound package mismatch')
 from controller import atomic
 atomic(EXP/'package/exp089-repair.tar.gz',raw)
if __name__=='__main__':main()
