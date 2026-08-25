#!/usr/bin/env python3
"""Evaluate detector-ensemble consensus as component repair proposals."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import torch

def load(name,file):
 s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file)); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def candidates(base,data):
 values=[]
 for axis in range(3):
  state=data['directions'].clone(); state[:,axis]=data['ensemble_consensus_directions'][:,axis]
  values.append(base.execute(state))
 values=torch.stack(values,1); initial=base.execute(data['directions'])
 error=lambda x: (x-data['targets'][:,None,:]).abs()[...,0]/5+(x-data['targets'][:,None,:]).abs()[...,1]/3
 gain=((initial-data['targets']).abs()[:,0]/5+(initial-data['targets']).abs()[:,1]/3)[:,None]-error(values)
 return gain,values

def choose(base,data,values,axis,edit):
 initial=base.execute(data['directions']); selected=values[torch.arange(len(axis)),axis]
 return torch.where(edit[:,None],selected,initial)

def calibrate(base,risk,learned,data,gain,values,score,limit):
 value,axis=score.float().max(1); rows=[]
 ts=torch.cat([torch.tensor([value.min()-1e-3,value.max()+1e-3]),torch.quantile(value,torch.linspace(0,1,41))]).unique()
 for t in ts:
  edit=value>t; pred=choose(base,data,values,axis,edit)
  row=risk.detailed_metrics(base,learned,data,pred,edit,gain,axis,.02); rows.append({'threshold':float(t),**row})
 feasible=[r for r in rows if r['joint_harm_rate']<=limit]
 return max(feasible,key=lambda r:(r['mean_protocol_gain'],r['opportunity_recall']))

def main():
 p=argparse.ArgumentParser(); p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--risk-limit',type=float,default=.10)
 p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/ensemble_consensus_repair_cv.json'));a=p.parse_args()
 base=load('ec_base','decisive_structured_refinement_benchmark.py');risk=load('ec_risk','risk_adjusted_selector_cv.py');learned=load('ec_learn','learned_proposal_selective_repair_benchmark.py');cf=load('ec_cf','crossfit_risk_selector_cv.py')
 rows=base.load_real_errors(a.results,a.annotations,a.image_dir);folds=[]
 for outer in range(a.folds):
  _,cr,tr=cf.outer_partitions(rows,outer,a.folds,a.seed);cal,test=cf.as_data(base,cr),cf.as_data(base,tr)
  cg,cv=candidates(base,cal);tg,tv=candidates(base,test)
  cs=cal['ensemble_axis_features'][...,1];ts=test['ensemble_axis_features'][...,1]
  policy=calibrate(base,risk,learned,cal,cg,cv,cs,a.risk_limit);value,axis=ts.float().max(1);edit=value>policy['threshold']
  oracle_axis,oracle_gain=tg.argmax(1),tg.max(1).values;oracle_edit=oracle_gain>0
  all_pred=base.execute(test['ensemble_consensus_directions']);initial=base.execute(test['directions']);dummy=axis
  metric=lambda pred,e,ax:risk.detailed_metrics(base,learned,test,pred,e,tg,ax,.02)
  methods={'no_repair':metric(initial,torch.zeros(len(axis),dtype=torch.bool),dummy),'consensus_repair_all':metric(all_pred,torch.ones(len(axis),dtype=torch.bool),dummy),'selective_consensus':metric(choose(base,test,tv,axis,edit),edit,axis),'oracle_selector_consensus':metric(choose(base,test,tv,oracle_axis,oracle_edit),oracle_edit,oracle_axis)}
  folds.append({'fold':outer,'methods':methods});a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'folds':folds,'aggregate':cf.aggregate(folds)},indent=2)+'\n');print(json.dumps({'fold':outer,'methods':methods}),flush=True)
 result={'folds':folds,'aggregate':cf.aggregate(folds)};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
