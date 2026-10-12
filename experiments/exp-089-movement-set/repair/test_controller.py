import copy,hashlib,importlib.util,json,math,multiprocessing,pathlib,subprocess,sys,tarfile,tempfile,unittest
EXP=pathlib.Path(__file__).resolve().parents[1];PATH=EXP/'repair/controller.py';s=importlib.util.spec_from_file_location('repair_controller',PATH);c=importlib.util.module_from_spec(s);s.loader.exec_module(c);DATA=json.loads((EXP/'repair/synthetic-input.json').read_text())
def cli(q):q.put(subprocess.run([sys.executable,str(PATH),'synthetic'],capture_output=True,text=True).returncode)
class TestController(unittest.TestCase):
 def badrow(self,edit,exc=ValueError):x=copy.deepcopy(DATA['rows']);edit(x);self.assertRaises(exc,c.validate_rows,x)
 def test_public_full_route(self):self.assertEqual(c.synthetic_route(load=0)['status'],'complete')
 def test_real_disabled_before_read(self):self.assertRaisesRegex(RuntimeError,'BEFORE FIRST REAL-VALUE READ',c.real_route)
 def test_load_hold_no_state_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   route=c.Route(d,c.FakeRemote(pathlib.Path(d)/'r'),c.Trace(c.FILES.values()));self.assertEqual(route.execute(11),{'status':'hold'});self.assertFalse(route.journal.exists())
 def test_two_processes(self):
  q=multiprocessing.Queue();a=multiprocessing.Process(target=cli,args=(q,));b=multiprocessing.Process(target=cli,args=(q,));a.start();b.start();a.join();b.join();self.assertEqual(sorted([q.get(),q.get()]),[0,0])
 def test_source_trace_complete(self):self.assertEqual(len(c.synthetic_route(load=0)['result']['trace']['opens']),4)
 def test_trace_omission(self):
  t=c.Trace(list(c.FILES.values())[:-1]);[t.read(x) for x in list(c.FILES.values())[:-1]];t.module('json');self.assertEqual(len(t.finish()['opens']),3)
 def test_trace_extra_fails(self):
  t=c.Trace(c.FILES.values());self.assertRaises(RuntimeError,t.read,EXP/'SPEC.md')
 def test_trace_unused_fails(self):self.assertRaises(RuntimeError,c.Trace(c.FILES.values()).finish)
 def test_unlisted_import_fails(self):
  t=c.Trace(c.FILES.values());t.module('math');self.assertRaises(RuntimeError,t.finish)
 def test_review_hash(self):self.assertEqual(c.sha(c.REVIEW),c.REVIEW_SHA)
 def test_aliases(self):
  for name in ('taxi_out_seconds','fold_error','p22','model_output','TARGET','NestedPredictionValue','Residual_Alias'):
   with self.subTest(name=name):self.assertRaises(ValueError,c.reject_columns,{name:1})
 def test_nested_alias(self):self.assertRaises(ValueError,c.reject_columns,{'ok':[{'TaXi_Out_Seconds':1}]})
 def test_query_label_mask(self):self.assertFalse(any('synthetic' in k for x in c.graph(DATA['rows']) for k in x))
 def test_neighbour_label_mask(self):self.assertTrue(all(len(x['features'])==c.MAX_DEGREE for x in c.graph(DATA['rows'])))
 def test_arrival_uses_destination(self):self.assertEqual(c.airport(DATA['rows'][1]),'AAA')
 def test_departure_uses_origin(self):self.assertEqual(c.airport(DATA['rows'][0]),'AAA')
 def test_missing_departure_airport(self):self.badrow(lambda x:x[0].update(ADEP_mvt=None))
 def test_missing_arrival_airport(self):self.badrow(lambda x:x[1].update(ADES_mvt=None))
 def test_bad_phase(self):self.badrow(lambda x:x[0].update(PHASE_mvt='X'))
 def test_schema_omission(self):self.badrow(lambda x:x[0].pop('feature'))
 def test_schema_addition(self):self.badrow(lambda x:x[0].update(extra=1))
 def test_native_id(self):self.badrow(lambda x:x[0].update(MVT_ID_mvt=1),TypeError)
 def test_duplicate_id(self):self.badrow(lambda x:x[1].update(MVT_ID_mvt=1.0))
 def test_ordered_id(self):self.badrow(lambda x:x.reverse())
 def test_nonfinite_input(self):self.badrow(lambda x:x[0].update(feature=float('nan')))
 def test_utc_required(self):self.badrow(lambda x:x[0].update(MVT_TIME_UTC_mvt='2025-01-01T00:00:00'))
 def test_fixed_degree(self):self.assertTrue(all(len(x['mask'])==32 for x in c.graph(DATA['rows'])))
 def test_exact_padding(self):self.assertTrue(all(sum(x['mask'])==x['degree_after_cap'] and all(v==0 for v,m in zip(x['features'],x['mask']) if not m) for x in c.graph(DATA['rows'])))
 def test_reliability_fields(self):self.assertTrue(all({'degree_before_cap','degree_after_cap','missing_field_fraction','time_span_seconds'}<=set(x) for x in c.graph(DATA['rows'])))
 def test_query_removed(self):self.assertTrue(all(x['query_id'] not in [i for i,m in zip(x['ids'],x['mask']) if m] for x in c.graph(DATA['rows'])))
 def test_controls_preserve_degree(self):
  g=c.graph(DATA['rows']);a,b=c.controls(g);self.assertEqual([x['degree_after_cap'] for x in g],[x['degree_after_cap'] for x in a]);self.assertEqual([x['degree_after_cap'] for x in g],[x['degree_after_cap'] for x in b])
 def test_controls_preserve_marginal(self):
  g=c.graph(DATA['rows']);_,b=c.controls(g);self.assertEqual(sorted(v for x in g for v,m in zip(x['features'],x['mask']) if m),sorted(v for x in b for v,m in zip(x['features'],x['mask']) if m))
 def test_capacity_actual(self):
  counts=[m.parameter_count for m in c.models().values()];self.assertLessEqual(max(counts)/min(counts),1.1)
 def test_effective_threads(self):self.assertEqual(set(c.effective_threads().values()),{'3'})
 def test_standard_json(self):self.assertRaises(ValueError,c.blob,float('nan'))
 def test_atomic_output(self):
  with tempfile.TemporaryDirectory() as d:p=pathlib.Path(d)/'x';c.atomic(p,{'x':1});self.assertEqual(json.loads(p.read_text()),{'x':1});self.assertFalse(list(pathlib.Path(d).glob('x.*')))
 def test_every_transition(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);c.Route(d,c.FakeRemote(d/'remote'),c.Trace(c.FILES.values())).execute(0);self.assertTrue(all(json.loads((d/f'{x}.json').read_text()) is not None for x in c.STEPS))
 def test_result_mutation(self):
  x=c.pilot(DATA);x['arms'].pop('row_only');self.assertRaises(ValueError,c.validate_result,x)
 def test_result_nonfinite(self):
  x=c.pilot(DATA);x['arms']['row_only']['gains'][0]=float('nan');self.assertRaises(ValueError,c.validate_result,x)
 def test_remote_identity_mutation(self):
  with tempfile.TemporaryDirectory() as d:
   r=c.FakeRemote(d,mutate='dataset_verified');receipt=r.action('dataset_verified',{});self.assertNotEqual(receipt['payload_sha256'],c.sha_bytes(c.blob({})))
 def test_crash_after_receipt_reconciles_without_replay(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);r=c.FakeRemote(d/'remote',crash='dataset_published:after');route=c.Route(d/'state',r,c.Trace(c.FILES.values()));self.assertRaises(SystemExit,route.execute,0);calls=len(r.calls);r.crash=None;route.execute(0);self.assertEqual(r.calls.count('dataset_published'),1);self.assertGreater(len(r.calls),calls)
 def test_missing_receipt_fails(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);route=c.Route(d,c.FakeRemote(d/'r'),c.Trace(c.FILES.values()));route.write({'step':'package_built','pending':'dataset_published','sequence':2});self.assertRaises(RuntimeError,route.execute,0)
 def test_first_failure_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   d=pathlib.Path(d);route=c.Route(d,c.FakeRemote(d/'r'),c.Trace([]));route.fail(ValueError('one'));first=route.failure.read_bytes();route.fail(ValueError('two'));self.assertEqual(route.failure.read_bytes(),first)
 def test_package_deterministic(self):
  bp=EXP/'repair/build_package.py';s=importlib.util.spec_from_file_location('builder',bp);b=importlib.util.module_from_spec(s);s.loader.exec_module(b);members={'a':b'a','b':b'b'};a,_=b.assemble(members,'fixed');z,_=b.assemble(members,'fixed');self.assertEqual(a,z)
if __name__=='__main__':unittest.main()
