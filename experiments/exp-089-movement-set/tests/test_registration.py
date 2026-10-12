import copy,importlib.util,json,pathlib,tempfile,unittest
EXP=pathlib.Path(__file__).resolve().parents[1];s=importlib.util.spec_from_file_location('ctl',EXP/'code/exp089_controller.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c);GOOD=json.loads((EXP/'fixtures.json').read_text())['rows']
class T(unittest.TestCase):
 def bad(self,f,e=ValueError):x=copy.deepcopy(GOOD);f(x);self.assertRaises(e,c.build_relations,x)
 def test_fixture(self):self.assertEqual(c.synthetic_fixture(EXP/'fixtures.json')[0]['schema'],1)
 def test_empty(self):self.assertRaises(ValueError,c.build_relations,[])
 def test_omit(self):self.bad(lambda x:x[0].pop('x'))
 def test_extra(self):self.bad(lambda x:x[0].update(extra=1))
 def test_id_type(self):self.bad(lambda x:x[0].update(MVT_ID_mvt=1),TypeError)
 def test_duplicate(self):self.bad(lambda x:x[1].update(MVT_ID_mvt=1.0))
 def test_order(self):self.bad(lambda x:x.reverse())
 def test_nan_id(self):self.bad(lambda x:x[0].update(MVT_ID_mvt=float('nan')))
 def test_inf(self):self.bad(lambda x:x[0].update(x=float('inf')))
 def test_naive_date(self):self.bad(lambda x:x[0].update(MVT_TIME_UTC_mvt='2025-01-01T00:00:00'))
 def test_non_utc(self):self.bad(lambda x:x[0].update(MVT_TIME_UTC_mvt='2025-01-01T01:00:00+01:00'))
 def test_label_alias(self):self.assertRaises(ValueError,c.reject_forbidden,['Taxi-Time sec'])
 def test_label_derivative(self):self.assertRaises(ValueError,c.reject_forbidden,['residual_feature'])
 def test_prediction(self):self.assertRaises(ValueError,c.reject_forbidden,['old_prediction_value'])
 def test_score(self):self.assertRaises(ValueError,c.reject_forbidden,['leaderboard_score'])
 def test_query_removed(self):self.assertTrue(all(x['query_id'] not in x['neighbour_ids'] for x in c.build_relations(GOOD)))
 def test_mask(self):self.assertTrue(all(len(x['mask'])==32 and sum(x['mask'])==x['degree_after_cap'] for x in c.build_relations(GOOD)))
 def test_cap(self):
  x=[dict(GOOD[0],MVT_ID_mvt=float(i+1),MVT_TIME_UTC_mvt=f'2025-07-01T10:{i:02d}:00+00:00') for i in range(40)];self.assertTrue(all(r['degree_after_cap']<=32 for r in c.build_relations(x)))
 def test_density(self):self.assertEqual([c.density_stratum(x) for x in [0,1,4,5,16,17,32,33]],['0','1-4','1-4','5-16','5-16','17-32','17-32','over-32-before-cap'])
 def test_degree_control(self):
  g=c.build_relations(GOOD);a,_=c.relation_controls(g);self.assertEqual([x['degree_after_cap'] for x in g],[x['degree_after_cap'] for x in a])
 def test_marginal_control(self):
  g=c.build_relations(GOOD);_,b=c.relation_controls(g);self.assertEqual([sorted(x['features']) for x in g],[sorted(x['features']) for x in b])
 def test_controls_fixed(self):self.assertEqual(c.relation_controls(c.build_relations(GOOD)),c.relation_controls(c.build_relations(GOOD)))
 def test_capacity(self):self.assertEqual(set(c.parameter_counts()),{'row_only','pooled_context','set_encoder'})
 def test_atomic(self):
  with tempfile.TemporaryDirectory() as d:p=pathlib.Path(d)/'x';c.atomic_write(p,{'a':1});self.assertEqual(json.loads(p.read_text()),{'a':1});self.assertFalse(list(pathlib.Path(d).glob('x.*')))
 def test_transitions(self):self.assertEqual(len(c.TRANSITIONS),9)
 def test_invalid_transition(self):
  old=c.JOURNAL
  with tempfile.TemporaryDirectory() as d:c.JOURNAL=pathlib.Path(d)/'j';self.assertRaises(RuntimeError,c.Controller().transition,'kernel_launched')
  c.JOURNAL=old
 def test_crash_recovery(self):
  oj,of=c.JOURNAL,c.FAILURE
  with tempfile.TemporaryDirectory() as d:
   c.JOURNAL=pathlib.Path(d)/'j';c.FAILURE=pathlib.Path(d)/'f';c.atomic_write(c.JOURNAL,{'state':'registered','sequence':0,'pending':'accepted'});self.assertRaises(RuntimeError,c.Controller().recover);self.assertEqual(c.Controller().state()['state'],'failed')
  c.JOURNAL,c.FAILURE=oj,of
 def test_threads(self):self.assertEqual(set(c.verify_threads().values()),{'3'})
 def test_disabled(self):self.assertFalse(c.ACCEPTANCE.exists())
 def test_cli_path_refused(self):self.assertRaises(SystemExit,c.main,['run','--path','x'])
 def test_result_forbids_candidate(self):self.assertIn('candidate',json.loads((EXP/'result-schema.json').read_text())['forbidden'])
 def test_one_repair(self):self.assertEqual(json.loads((EXP/'phase-a-spec.json').read_text())['repair']['family'],'density_or_reliability_only')
 def test_no_board_source(self):self.assertFalse(any('board/' in p for p in json.loads((EXP/'source-manifest.json').read_text())['sources']))
 def test_remote_identity_fixed(self):self.assertEqual(c.main.__defaults__,(None,))
if __name__=='__main__':unittest.main()
