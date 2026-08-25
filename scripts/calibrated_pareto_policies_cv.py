#!/usr/bin/env python3
"""Calibrate gate and shrinkage under magnitude-aware harm constraints."""
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

def pred(base,data,candidates,score,coverage,alpha,space,all_cases=False):
 initial=base.execute(data['directions'])
 if all_cases:
  edit=torch.full((len(initial),),alpha>0,dtype=torch.bool)
  if alpha==0:return initial,edit
  if space=='output':return initial+alpha*(base.execute(data['ensemble_consensus_directions'])-initial),edit
  directions=torch.nn.functional.normalize(data['directions']+alpha*(data['ensemble_consensus_directions']-data['directions']),dim=-1);return base.execute(directions),edit
 axis,edit=top(score,coverage);selected=candidates[torch.arange(len(axis)),axis]
 if space=='output':return torch.where(edit[:,None],initial+alpha*(selected-initial),initial),edit&(alpha>0)
 directions=data['directions'].clone();current=directions[torch.arange(len(axis)),axis];proposal=data['ensemble_consensus_directions'][torch.arange(len(axis)),axis];blended=torch.nn.functional.normalize(current+alpha*(proposal-current),dim=-1);directions[torch.arange(len(axis))[edit],axis[edit]]=blended[edit];return base.execute(directions),edit&(alpha>0)

def metrics(base,data,prediction,edit):
 before=(base.execute(data['directions'])-data['targets']).abs().mean(1);after=(prediction-data['targets']).abs().mean(1);w=after-before
 return {'mean_MAE':float(after.mean()),'P90_MAE':float(torch.quantile(after,.9)),'P95_MAE':float(torch.quantile(after,.95)),'trigger_rate':float(edit.float().mean()),'harm_any':float((w>0).float().mean()),'harm_at_0.5deg':float((w>.5).float().mean()),'harm_at_1deg':float((w>1).float().mean()),'harm_at_2deg':float((w>2).float().mean())}

def select(base,data,candidates,score,family,tau,epsilon):
 selector,space=family.split('_');coverages=[1.] if selector=='all' else [0,.05,.1,.2,.3,.4,.6,.8,1.];alphas=[0,.1,.2,.3,.4,.5,.7,1.];rows=[]
 for coverage in coverages:
  for alpha in alphas:
   prediction,edit=pred(base,data,candidates,score,coverage,alpha,space,selector=='all');row=metrics(base,data,prediction,edit);rows.append({'coverage':coverage,'alpha':alpha,**row})
 feasible=[row for row in rows if row[f'harm_at_{tau:g}deg']<=epsilon];return min(feasible,key=lambda row:(row['mean_MAE'],row['harm_any']))

def main():
 p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/calibrated_pareto_policies_cv.json'));a=p.parse_args()
 base=load('cp_base','decisive_structured_refinement_benchmark.py');sel=load('cp_sel','selective_axis_verifier_benchmark.py');cf=load('cp_cf','crossfit_risk_selector_cv.py');cons=load('cp_cons','ensemble_consensus_repair_cv.py');eg=load('cp_eg','ensemble_gain_model_cv.py');rows=base.load_real_errors(a.results,a.annotations,a.image_dir);eg.configure_ensemble(base,rows,3);folds=[];families=['all_geometry','all_output','uncertainty_geometry','sensitivity_geometry','learned_geometry','learned_output'];objectives=[(.5,.01),(.5,.02),(1.,.01)]
 for outer in range(a.folds):
  rr,cr,tr=cf.outer_partitions(rows,outer,a.folds,a.seed);train,cal,test=[cf.as_data(base,x) for x in [rr,cr,tr]];train_gain,_=cons.candidates(base,train);cal_gain,cal_candidates=cons.candidates(base,cal);test_gain,test_candidates=cons.candidates(base,test);model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,random_state=42).fit(eg.features(base,sel,train,'full'),train_gain.reshape(-1).numpy());cal_scores={'learned':eg.predict(model,eg.features(base,sel,cal,'full'),len(cal['targets'])),'uncertainty':cal['ensemble_axis_features'][...,1],'sensitivity':sel.sensitivities(base,cal['directions']),'all':torch.zeros(len(cal['targets']),3)};test_scores={'learned':eg.predict(model,eg.features(base,sel,test,'full'),len(test['targets'])),'uncertainty':test['ensemble_axis_features'][...,1],'sensitivity':sel.sensitivities(base,test['directions']),'all':torch.zeros(len(test['targets']),3)};methods={}
  initial=base.execute(test['directions']);methods['no_repair']=metrics(base,test,initial,torch.zeros(len(initial),dtype=torch.bool))
  for tau,epsilon in objectives:
   for family in families:
    selector=family.split('_')[0];policy=select(base,cal,cal_candidates,cal_scores[selector],family,tau,epsilon);prediction,edit=pred(base,test,test_candidates,test_scores[selector],policy['coverage'],policy['alpha'],family.split('_')[1],selector=='all');methods[f'{family}|tau={tau:g}|epsilon={epsilon:g}']={**metrics(base,test,prediction,edit),'coverage_policy':policy['coverage'],'alpha_policy':policy['alpha']}
  folds.append({'fold':outer,'methods':methods});print(json.dumps({'fold':outer,'methods':methods}),flush=True)
 result={'patient_grouped':True,'calibration_only_policy_selection':True,'objectives':[{'tau':tau,'epsilon':epsilon} for tau,epsilon in objectives],'folds':folds,'aggregate':cf.aggregate(folds)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
