#!/usr/bin/env python3
"""Matched selector baselines for one-axis ensemble-consensus proposals."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch
from sklearn.ensemble import ExtraTreesRegressor

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def top_budget(scores,fraction):
 value,axis=scores.max(1);k=max(0,min(len(value),int(round(fraction*len(value)))));edit=torch.zeros(len(value),dtype=torch.bool)
 if k: edit[value.topk(k).indices]=True
 return axis,edit

def evaluate(base,risk,learned,cons,data,gain,candidates,scores,fraction):
 axis,edit=top_budget(scores,fraction);pred=cons.choose(base,data,candidates,axis,edit)
 return risk.detailed_metrics(base,learned,data,pred,edit,gain,axis,.02)

def evaluate_with_cases(base,risk,learned,cons,data,gain,candidates,scores,fraction):
 axis,edit=top_budget(scores,fraction);pred=cons.choose(base,data,candidates,axis,edit);metric=risk.detailed_metrics(base,learned,data,pred,edit,gain,axis,.02);before=(base.execute(data['directions'])-data['targets']).abs();after=(pred-data['targets']).abs()
 cases=[{'identifier':identifier,'mean_before':float(b.mean()),'mean_after':float(c.mean()),'edited':bool(e)} for identifier,b,c,e in zip(data['identifiers'],before,after,edit)]
 return metric,cases

def random_average(base,risk,learned,cons,data,gain,candidates,fraction,seed,repeats=100):
 rows=[];cases=[]
 for repeat in range(repeats):
  generator=torch.Generator().manual_seed(seed+repeat);scores=torch.rand(len(data['targets']),3,generator=generator)
  metric,current=evaluate_with_cases(base,risk,learned,cons,data,gain,candidates,scores,fraction);rows.append(metric);cases.extend([dict(row,repeat=repeat) for row in current])
 return {key:float(np.mean([row[key] for row in rows])) for key in rows[0]},cases

def main():
 p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--coverage',type=float,default=.20)
 p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/selector_baselines_fixed20.json'));a=p.parse_args()
 base=load('sb_base','decisive_structured_refinement_benchmark.py');sel=load('sb_sel','selective_axis_verifier_benchmark.py');learned=load('sb_l','learned_proposal_selective_repair_benchmark.py');risk=load('sb_r','risk_adjusted_selector_cv.py');cf=load('sb_cf','crossfit_risk_selector_cv.py');cons=load('sb_c','ensemble_consensus_repair_cv.py');eg=load('sb_eg','ensemble_gain_model_cv.py')
 rows=base.load_real_errors(a.results,a.annotations,a.image_dir);eg.configure_ensemble(base,rows,3);folds=[]
 for outer in range(a.folds):
  rr,_,tr=cf.outer_partitions(rows,outer,a.folds,a.seed);train,test=cf.as_data(base,rr),cf.as_data(base,tr);train_gain,_=cons.candidates(base,train);gain,candidates=cons.candidates(base,test)
  model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,random_state=42).fit(eg.features(base,sel,train,'full'),train_gain.reshape(-1).numpy())
  learned_score=eg.predict(model,eg.features(base,sel,test,'full'),len(test['targets']));sensitivity=sel.sensitivities(base,test['directions']);displacement=(test['ensemble_consensus_directions']-test['directions']).norm(dim=-1);uncertainty=test['ensemble_axis_features'][...,1]
  random_metric,random_cases=random_average(base,risk,learned,cons,test,gain,candidates,a.coverage,100000+a.seed*10+outer);methods={'random':random_metric};cases={'random':random_cases}
  for name,score in {'largest_uncertainty':uncertainty,'largest_displacement':displacement,'largest_sensitivity':sensitivity,'learned_expected_gain':learned_score}.items():methods[name],cases[name]=evaluate_with_cases(base,risk,learned,cons,test,gain,candidates,score,a.coverage)
  initial=base.execute(test['directions']);dummy=torch.zeros(len(initial),dtype=torch.long);methods['no_repair']=risk.detailed_metrics(base,learned,test,initial,torch.zeros(len(initial),dtype=torch.bool),gain,dummy,.02)
  oracle_axis,oracle_gain=gain.max(1);oracle_component=gain.argmax(1);oracle_edit=oracle_axis>0;methods['oracle']=risk.detailed_metrics(base,learned,test,cons.choose(base,test,candidates,oracle_component,oracle_edit),oracle_edit,gain,oracle_component,.02)
  folds.append({'fold':outer,'methods':methods,'cases':cases});print(json.dumps({'fold':outer,'methods':methods}),flush=True)
 result={'patient_grouped':True,'coverage_budget':a.coverage,'split_seed':a.seed,'folds':folds,'aggregate':cf.aggregate(folds)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
