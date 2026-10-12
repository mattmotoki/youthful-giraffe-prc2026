import copy,hashlib,importlib.util,json,math,pathlib,tarfile,tempfile,unittest
EXP=pathlib.Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('ctl',EXP/'code/controller.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
GOOD=json.loads((EXP/'fixtures/synthetic-v2.json').read_text()); C=c.contract()
class RepairTests(unittest.TestCase):
 def test_fixture(self):self.assertEqual(c.fixture(EXP/'fixtures/synthetic-v2.json')[0],GOOD)
 def bad(self,edit):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'x';x=copy.deepcopy(GOOD);edit(x);p.write_text(json.dumps(x));return self.assertRaises(ValueError,c.fixture,p)
 def test_empty_a(self):self.bad(lambda x:x.__setitem__('a_rows',[]))
 def test_empty_features(self):self.bad(lambda x:x.__setitem__('feature_rows',[]))
 def test_id_dtype(self):self.bad(lambda x:x['a_rows'][0].__setitem__('MVT_ID_mvt','101'))
 def test_duplicate_id(self):self.bad(lambda x:x['a_rows'][1].__setitem__('MVT_ID_mvt',101))
 def test_duplicate_key(self):self.bad(lambda x:x['feature_rows'][1].__setitem__('key','s0'))
 def test_negative_count(self):self.bad(lambda x:x['feature_rows'][0].__setitem__('n_A',-1))
 def test_nonfinite(self):self.bad(lambda x:x['a_rows'][0].__setitem__('x',float('nan')))
 def test_bad_date(self):self.bad(lambda x:x['feature_rows'][0].__setitem__('utc_date','bad'))
 def test_extra_field(self):self.bad(lambda x:x['a_rows'][0].__setitem__('extra',1))
 def test_supplied_null_refused(self):self.bad(lambda x:x.__setitem__('null_rows',[{}]))
 def test_blocks_one(self):self.assertEqual(len(c.blocks(GOOD['feature_rows'],1)),6)
 def test_blocks_three(self):self.assertEqual(c.blocks(GOOD['feature_rows'],3)[0],[0,1,2])
 def test_block_gap(self):
  x=copy.deepcopy(GOOD['feature_rows'])
  for i,r in enumerate(x):r['utc_date']=f'2025-04-{1+2*i:02d}'
  self.assertRaises(ValueError,c.blocks,x,3)
 def test_null_ten_shams(self):self.assertEqual(len(c.null_record(GOOD['feature_rows'],C)['selected']),10)
 def test_null_all_cells(self):self.assertEqual(len(c.null_record(GOOD['feature_rows'],C)['attempts'][0]['cells']),72)
 def test_null_same_permutation_across_c(self):
  a=c.null_record(GOOD['feature_rows'],C)['attempts'][0]['cells'];self.assertEqual(len({tuple(x['permutation']) for x in a if x['family']=='stand' and x['threshold']==25}),1)
 def test_null_multiset(self):self.assertTrue(all(x['accepted'] for x in c.null_record(GOOD['feature_rows'],C)['attempts'] if x['accepted']))
 def test_null_missing_group(self):self.assertRaises(ValueError,c.null_record,GOOD['feature_rows'][:-2],C)
 def test_contract_rotation(self):self.assertEqual(C['rotation'],{'V':'2025-05','S':'2025-06','I':'2025-04'})
 def test_contract_endpoint(self):self.assertFalse(C['endpoint']['board_recombination'])
 def test_standard_json(self):self.assertRaises(ValueError,c.blob,float('nan'))
 def test_atomic(self):
  with tempfile.TemporaryDirectory() as d:
   p=pathlib.Path(d)/'x';c.atomic(p,{'x':1});self.assertEqual(json.loads(p.read_text()),{'x':1});self.assertFalse(list(pathlib.Path(d).glob('x.*')))
 def test_two_builds(self):
  a,_=c.build('81fba5cb29c5cb9ba0857d9e42288ede1da83295');b,_=c.build('81fba5cb29c5cb9ba0857d9e42288ede1da83295');self.assertEqual(a,b)
 def test_package_identity(self):
  p=EXP/'package-v2/exp088-repair.bin';self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),'592d58923400f4bd13a18093d709eb79107c21220bd4f6cc2575a58985b57358')
 def test_package_members(self):
  with tarfile.open(EXP/'package-v2/exp088-repair.bin','r:gz') as t:self.assertTrue(all(x.isfile() for x in t.getmembers()));self.assertEqual(len(t.getnames()),len(set(t.getnames())))
 def test_threads_effective(self):
  x=json.loads((EXP/'private-probe-v2/output/transport-result-v2.json').read_text())['threads'];self.assertEqual(x['lightgbm'],{'n_jobs':3,'num_threads':3});self.assertEqual(x['catboost']['thread_count'],3);self.assertTrue(all(v=='3' for v in x['environment_subprocess'].values()))
 def test_remote_output_package(self):
  x=json.loads((EXP/'private-probe-v2/output/transport-result-v2.json').read_text());self.assertTrue(x['pass']);self.assertEqual(x['package_sha256'],'592d58923400f4bd13a18093d709eb79107c21220bd4f6cc2575a58985b57358')
 def test_remote_mutation(self):
  x=json.loads((EXP/'private-probe-v2/output/transport-result-v2.json').read_text());x['package_sha256']='0'*64;self.assertNotEqual(x['package_sha256'],c.sha(EXP/'package-v2/exp088-repair.bin'))
 def test_registration_exact(self):self.assertEqual(json.loads((EXP/'source-manifest.json').read_text())['registration_commit'],'349cbe4fbfaccc087c9891eaede4a3303376c56f')
 def test_future_acceptance_absent(self):self.assertFalse((EXP/'review-acceptance.json').exists())
 def test_disabled_boundary(self):self.assertNotIn('label',GOOD['feature_rows'][0])
if __name__=='__main__':unittest.main()
