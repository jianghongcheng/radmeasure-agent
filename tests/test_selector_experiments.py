import importlib.util
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def test_top_budget_has_exact_rounded_case_coverage():
 selector=module('selector_baselines_cv');scores=torch.arange(30,dtype=torch.float32).reshape(10,3);axis,edit=selector.top_budget(scores,.2)
 assert edit.sum().item()==2
 assert axis.shape==(10,)

def test_single_detector_consensus_is_a_geometric_noop():
 experiment=module('ensemble_gain_model_cv')
 class Base:
  @staticmethod
  def canonical(x):return torch.nn.functional.normalize(x,dim=-1)
 direction=torch.tensor([[1.,0.],[0.,1.],[1.,1.]])
 rows=[{'identifier':'x','seed':seed,'predicted_directions':torch.roll(direction,i,0)} for i,seed in enumerate([17,42,73])]
 experiment.configure_ensemble(Base,rows,1)
 for row in rows:assert torch.allclose(row['ensemble_consensus_directions'],row['predicted_directions']/row['predicted_directions'].norm(dim=-1,keepdim=True))

def test_two_detector_consensus_changes_with_companion():
 experiment=module('ensemble_gain_model_cv')
 class Base:
  @staticmethod
  def canonical(x):return torch.nn.functional.normalize(x,dim=-1)
 rows=[{'identifier':'x','seed':seed,'predicted_directions':torch.tensor([[1.,float(i+1)],[1.,0.],[0.,1.]])} for i,seed in enumerate([17,42,73])]
 experiment.configure_ensemble(Base,rows,2,1);first=rows[0]['ensemble_consensus_directions'].clone();experiment.configure_ensemble(Base,rows,2,2);second=rows[0]['ensemble_consensus_directions']
 assert not torch.allclose(first,second)

def test_pair_expansion_contains_only_requested_pairing():
 pairing=module('pairing_generalization_cv')
 class Base:
  @staticmethod
  def canonical(x):return torch.nn.functional.normalize(x,dim=-1)
 rows=[{'identifier':'x','seed':seed,'predicted_directions':torch.tensor([[1.,float(i+1)],[1.,0.],[0.,1.]])} for i,seed in enumerate([17,42,73])]
 expanded=pairing.expand_pairs(Base,rows,[(17,73)])
 assert len(expanded)==2
 assert {row['seed'] for row in expanded}=={17,73}
 assert {row['pairing'] for row in expanded}=={'17-73'}

def test_zero_shrinkage_is_exact_stop():
 shrinkage=module('shrinkage_baseline_cv')
 class Base:
  @staticmethod
  def execute(x):return x[...,0]
 data={'directions':torch.tensor([[[1.,0.],[2.,0.]]]),'ensemble_consensus_directions':torch.tensor([[[3.,0.],[4.,0.]]])}
 assert torch.equal(shrinkage.prediction(Base,data,0.,'geometry'),Base.execute(data['directions']))
 assert torch.equal(shrinkage.prediction(Base,data,0.,'output'),Base.execute(data['directions']))

def test_output_shrinkage_is_convex_interpolation():
 shrinkage=module('shrinkage_baseline_cv')
 class Base:
  @staticmethod
  def execute(x):return x[...,0]
 data={'directions':torch.tensor([[[1.,0.],[2.,0.]]]),'ensemble_consensus_directions':torch.tensor([[[3.,0.],[4.,0.]]])}
 assert torch.allclose(shrinkage.prediction(Base,data,.5,'output'),torch.tensor([[2.,3.]]))

def test_magnitude_harm_ignores_subthreshold_worsening():
 analysis=module('harm_magnitude_analysis');rows=[{'mean_before':1.,'mean_after':1.1,'edited':True},{'mean_before':1.,'mean_after':1.6,'edited':True}]
 result=analysis.metrics(rows)
 assert result['harm_any']==1.
 assert result['harm_at_0.5deg']==.5
 assert result['harm_at_1deg']==0.

def test_adaptive_action_space_has_exact_counterfactual_labels():
 adaptive=module('adaptive_action_advantage_cv')
 class Base:
  @staticmethod
  def execute(x):return torch.stack([x[:,0,0],x[:,1,0]],1)
 class Features:
  @staticmethod
  def features(base,selector,data,mode):return torch.zeros(len(data['targets'])*3,2).numpy()
 data={'targets':torch.tensor([[1.,1.]]),'directions':torch.tensor([[[1.,0.],[1.,0.],[1.,0.]]]),'ensemble_consensus_directions':torch.tensor([[[0.,1.],[0.,1.],[0.,1.]]])}
 features,predictions,gain=adaptive.actions(Base,None,Features,data)
 assert features.shape[:3]==(1,3,len(adaptive.ALPHAS))
 assert predictions.shape==(1,3,len(adaptive.ALPHAS),2)
 before=(Base.execute(data['directions'])-data['targets']).abs().mean(1)
 expected=before[:,None,None]-(predictions-data['targets'][:,None,None]).abs().mean(-1)
 assert torch.allclose(gain,expected)
