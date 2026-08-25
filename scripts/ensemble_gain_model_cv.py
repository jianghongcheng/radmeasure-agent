#!/usr/bin/env python3
"""Tabular component gain models for deterministic ensemble-consensus proposals."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, HistGradientBoostingRegressor

def load(name,file):
 s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

FEATURE_MODES = ('full', 'no_uncertainty', 'no_sensitivity', 'geometry_only',
                 'uncertainty_only', 'no_axis_identity')

def features(base,selective,data,mode='full'):
 b=len(data['targets']); sens=selective.sensitivities(base,data['directions']); angles=base.execute(data['directions'])/90
 delta=data['ensemble_consensus_directions']-data['directions']; onehot=torch.eye(3)[None].expand(b,-1,-1)
 shared=torch.cat([angles,data['aspect']],1)[:,None].expand(-1,3,-1)
 groups={
  'uncertainty':data['ensemble_axis_features'], 'sensitivity':sens[...,None],
  'current_geometry':data['directions'], 'proposal_geometry':data['ensemble_consensus_directions'],
  'proposal_delta':delta, 'measurement_context':shared, 'axis_identity':onehot}
 selected={
  'full':tuple(groups),
  'no_uncertainty':tuple(k for k in groups if k!='uncertainty'),
  'no_sensitivity':tuple(k for k in groups if k!='sensitivity'),
  'geometry_only':('current_geometry','proposal_geometry','proposal_delta','measurement_context','axis_identity'),
  'uncertainty_only':('uncertainty','measurement_context','axis_identity'),
  'no_axis_identity':tuple(k for k in groups if k!='axis_identity')}[mode]
 x=torch.cat([groups[k] for k in selected],-1)
 return x.reshape(-1,x.shape[-1]).numpy()

def predict(model,x,n):return torch.tensor(model.predict(x),dtype=torch.float32).reshape(n,3)

def configure_ensemble(base,rows,size,companion_offset=1):
 """Build a per-base K-model consensus without leaking excluded detector seeds."""
 groups={}
 for row in rows: groups.setdefault(row['identifier'],[]).append(row)
 for group in groups.values():
  ordered=sorted(group,key=lambda r:r['seed'])
  for i,row in enumerate(ordered):
   if size==2: members=[row,ordered[(i+companion_offset)%len(ordered)]]
   else: members=[row]+[ordered[(i+j)%len(ordered)] for j in range(1,size)]
   stacked=torch.stack([member['predicted_directions'] for member in members])
   consensus=base.canonical(stacked.mean(0));deviations=torch.rad2deg(torch.acos((stacked*consensus[None]).sum(-1).abs().clamp(0,1-1e-7)))
   row['ensemble_consensus_directions']=consensus
   row['ensemble_axis_features']=torch.stack([deviations[0],deviations.mean(0)],-1)

def calibrate(base,risk,learned,cons,data,gain,candidates,scores,limit):
 value,axis=scores.max(1); ts=torch.cat([torch.tensor([value.min()-1e-3,value.max()+1e-3]),torch.quantile(value,torch.linspace(0,1,41))]).unique();rows=[]
 for t in ts:
  edit=value>t;pred=cons.choose(base,data,candidates,axis,edit);row=risk.detailed_metrics(base,learned,data,pred,edit,gain,axis,.02);rows.append({'threshold':float(t),**row})
 feasible=[r for r in rows if r['joint_harm_rate']<=limit];return max(feasible,key=lambda r:(r['mean_protocol_gain'],r['opportunity_recall']))

def main():
 p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--risk-limit',type=float,default=.10)
 p.add_argument('--feature-mode',choices=FEATURE_MODES,default='full')
 p.add_argument('--target-coverage',type=float,default=None,help='Label-free top-score intervention budget; bypasses harm-calibrated threshold.')
 p.add_argument('--ensemble-size',type=int,choices=(1,2,3),default=3)
 p.add_argument('--ensemble-companion-offset',type=int,choices=(1,2),default=1,help='Which independent detector accompanies the base when K=2.')
 p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/ensemble_gain_model_cv.json'));a=p.parse_args()
 base=load('eg_base','decisive_structured_refinement_benchmark.py');sel=load('eg_sel','selective_axis_verifier_benchmark.py');learned=load('eg_l','learned_proposal_selective_repair_benchmark.py');risk=load('eg_r','risk_adjusted_selector_cv.py');cf=load('eg_cf','crossfit_risk_selector_cv.py');cons=load('eg_c','ensemble_consensus_repair_cv.py')
 rows=base.load_real_errors(a.results,a.annotations,a.image_dir);configure_ensemble(base,rows,a.ensemble_size,a.ensemble_companion_offset);folds=[]
 factories={'random_forest':lambda:RandomForestRegressor(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,random_state=42),'extra_trees':lambda:ExtraTreesRegressor(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,random_state=42),'hist_gradient_boosting':lambda:HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,l2_regularization=1.,random_state=42)}
 for outer in range(a.folds):
  rr,cr,tr=cf.outer_partitions(rows,outer,a.folds,a.seed);train,cal,test=[cf.as_data(base,x) for x in [rr,cr,tr]];train_gain,_=cons.candidates(base,train);cal_gain,cal_c=cons.candidates(base,cal);test_gain,test_c=cons.candidates(base,test)
  xtr=features(base,sel,train,a.feature_mode);y=train_gain.reshape(-1).numpy();xc=features(base,sel,cal,a.feature_mode);xt=features(base,sel,test,a.feature_mode);methods={};case_rows={}
  for name,factory in factories.items():
   model=factory().fit(xtr,y);cs=predict(model,xc,len(cal['targets']));ts=predict(model,xt,len(test['targets']));value,axis=ts.max(1)
   if a.target_coverage is None:
    policy=calibrate(base,risk,learned,cons,cal,cal_gain,cal_c,cs,a.risk_limit);edit=value>policy['threshold'];threshold=policy['threshold']
   else:
    k=min(len(value),max(0,int(round(a.target_coverage*len(value)))));edit=torch.zeros(len(value),dtype=torch.bool)
    if k: edit[value.topk(k).indices]=True
    threshold=float(value[edit].min()) if k else float('inf')
   pred=cons.choose(base,test,test_c,axis,edit);methods[name]=risk.detailed_metrics(base,learned,test,pred,edit,test_gain,axis,.02);methods[name]['threshold']=threshold;before=(base.execute(test['directions'])-test['targets']).abs();after=(pred-test['targets']).abs();case_rows[name]=[{'identifier':identifier,'HVA_before':float(b[0]),'IMA_before':float(b[1]),'HVA_after':float(c[0]),'IMA_after':float(c[1]),'edited':bool(e)} for identifier,b,c,e in zip(test['identifiers'],before,after,edit)]
  initial=base.execute(test['directions']);dummy=test_gain.argmax(1);methods['no_repair']=risk.detailed_metrics(base,learned,test,initial,torch.zeros(len(dummy),dtype=torch.bool),test_gain,dummy,.02);oa,og=test_gain.argmax(1),test_gain.max(1).values;oe=og>0;methods['oracle']=risk.detailed_metrics(base,learned,test,cons.choose(base,test,test_c,oa,oe),oe,test_gain,oa,.02)
  metadata={'feature_mode':a.feature_mode,'risk_limit':a.risk_limit,'target_coverage':a.target_coverage,'ensemble_size':a.ensemble_size,'ensemble_companion_offset':a.ensemble_companion_offset}
  folds.append({'fold':outer,'methods':methods,'cases':case_rows});a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({**metadata,'folds':folds,'aggregate':cf.aggregate(folds)},indent=2)+'\n');print(json.dumps({'fold':outer,'methods':methods}),flush=True)
 result={**metadata,'folds':folds,'aggregate':cf.aggregate(folds)};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
