#!/usr/bin/env python3
"""Unified MAE-versus-magnitude-aware-harm frontier on patient-grouped folds."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch
from sklearn.ensemble import ExtraTreesRegressor

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def top(scores,fraction):
 value,axis=scores.max(1);k=int(round(fraction*len(value)));edit=torch.zeros(len(value),dtype=torch.bool)
 if k:edit[value.topk(k).indices]=True
 return axis,edit

def gated_prediction(base,data,candidates,axis,edit,alpha,space):
 initial=base.execute(data['directions']);selected=candidates[torch.arange(len(axis)),axis]
 if space=='output':return torch.where(edit[:,None],initial+alpha*(selected-initial),initial)
 directions=data['directions'].clone();chosen=directions[torch.arange(len(axis)),axis];proposal=data['ensemble_consensus_directions'][torch.arange(len(axis)),axis];blended=torch.nn.functional.normalize(chosen+alpha*(proposal-chosen),dim=-1);directions[torch.arange(len(axis))[edit],axis[edit]]=blended[edit];return base.execute(directions)

def all_prediction(base,data,alpha,space):
 initial=base.execute(data['directions'])
 if alpha==0:return initial
 if space=='output':return initial+alpha*(base.execute(data['ensemble_consensus_directions'])-initial)
 return base.execute(torch.nn.functional.normalize(data['directions']+alpha*(data['ensemble_consensus_directions']-data['directions']),dim=-1))

def add(store,key,data,pred,edit):
 before=(base_execute(data)-data['targets']).abs().mean(1);after=(pred-data['targets']).abs().mean(1)
 store.setdefault(key,[]).extend([{'identifier':identifier,'before':float(b),'after':float(c),'edited':bool(e)} for identifier,b,c,e in zip(data['identifiers'],before,after,edit)])

def base_execute(data):return EXECUTE(data['directions'])

def summarize(rows):
 before=np.array([r['before'] for r in rows]);after=np.array([r['after'] for r in rows]);w=after-before;result={'n':len(rows),'mean_MAE':float(after.mean()),'P90_MAE':float(np.quantile(after,.9)),'P95_MAE':float(np.quantile(after,.95)),'trigger_rate':float(np.mean([r['edited'] for r in rows])),'harm_any':float((w>0).mean())}
 for tau in [.5,1.,2.]:result[f'harm_at_{tau:g}deg']=float((w>tau).mean())
 return result

def clustered(rows):
 groups={}
 for row in rows:groups.setdefault(row['identifier'],[]).append(row)
 return [{'identifier':key,'before':float(np.mean([r['before'] for r in values])),'after':float(np.mean([r['after'] for r in values])),'edited':float(np.mean([r['edited'] for r in values]))} for key,values in groups.items()]

def main():
 global EXECUTE
 p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/pareto_frontier_cv.json'));a=p.parse_args()
 base=load('pf_base','decisive_structured_refinement_benchmark.py');sel=load('pf_sel','selective_axis_verifier_benchmark.py');cf=load('pf_cf','crossfit_risk_selector_cv.py');cons=load('pf_cons','ensemble_consensus_repair_cv.py');eg=load('pf_eg','ensemble_gain_model_cv.py');EXECUTE=base.execute;rows=base.load_real_errors(a.results,a.annotations,a.image_dir);eg.configure_ensemble(base,rows,3);store={};coverages=[0,.05,.1,.2,.3,.4,.6,.8,1.];alphas=[0,.1,.2,.3,.4,.5,.7,1.]
 for outer in range(a.folds):
  rr,_,tr=cf.outer_partitions(rows,outer,a.folds,a.seed);train,test=cf.as_data(base,rr),cf.as_data(base,tr);train_gain,_=cons.candidates(base,train);_,candidates=cons.candidates(base,test);model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,random_state=42).fit(eg.features(base,sel,train,'full'),train_gain.reshape(-1).numpy());scores={'learned':eg.predict(model,eg.features(base,sel,test,'full'),len(test['targets'])),'uncertainty':test['ensemble_axis_features'][...,1],'sensitivity':sel.sensitivities(base,test['directions'])}
  initial=base.execute(test['directions']);add(store,'no_repair',test,initial,torch.zeros(len(initial),dtype=torch.bool))
  for alpha in alphas:
   for space in ['geometry','output']:add(store,f'all_{space}|alpha={alpha}',test,all_prediction(base,test,alpha,space),torch.full((len(initial),),alpha>0,dtype=torch.bool))
  for selector_name,score in scores.items():
   for coverage in coverages:
    axis,edit=top(score,coverage)
    for alpha in alphas:
     for space in ['geometry','output']:
      pred=gated_prediction(base,test,candidates,axis,edit,alpha,space);add(store,f'{selector_name}_{space}|coverage={coverage}|alpha={alpha}',test,pred,edit&(alpha>0))
  print(json.dumps({'completed_fold':outer}),flush=True)
 points=[]
 for key,values in store.items():
  family,*settings=key.split('|');row={'family':family};row.update({name:float(value) for setting in settings for name,value in [setting.split('=')]});row['row_level']=summarize(values);row['identifier_clustered']=summarize(clustered(values));points.append(row)
 result={'patient_grouped':True,'harm_definition':'increase in per-case mean HVA/IMA absolute error by more than tau degrees','points':points};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'n_points':len(points)},indent=2))
if __name__=='__main__':main()
