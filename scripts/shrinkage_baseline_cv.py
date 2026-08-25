#!/usr/bin/env python3
"""Patient-grouped soft-update baselines against binary selective correction."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import torch

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def prediction(base,data,alpha,space):
 initial=base.execute(data['directions']);consensus=data['ensemble_consensus_directions']
 if alpha==0:return initial
 if space=='output':return initial+alpha*(base.execute(consensus)-initial)
 directions=torch.nn.functional.normalize(data['directions']+alpha*(consensus-data['directions']),dim=-1);return base.execute(directions)

def row(base,learned,data,alpha,space):
 edit=torch.full((len(data['targets']),),alpha>0,dtype=torch.bool);result=learned.metrics(base,data,prediction(base,data,alpha,space),edit);result['alpha']=float(alpha);return result

def cases(base,data,alpha,space):
 initial=base.execute(data['directions']);pred=prediction(base,data,alpha,space);before=(initial-data['targets']).abs().mean(1);after=(pred-data['targets']).abs().mean(1)
 return [{'identifier':identifier,'mean_before':float(b),'mean_after':float(c),'edited':alpha>0} for identifier,b,c in zip(data['identifiers'],before,after)]

def choose(base,learned,cal,space,risk=None):
 rows=[row(base,learned,cal,float(alpha),space) for alpha in torch.linspace(0,1,21)];feasible=rows if risk is None else [item for item in rows if item['joint_harm_rate']<=risk];return min(feasible,key=lambda item:(item['mean_MAE'],item['joint_harm_rate']))['alpha']

def main():
 p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/shrinkage_baseline_cv.json'));a=p.parse_args()
 base=load('sh_base','decisive_structured_refinement_benchmark.py');learned=load('sh_l','learned_proposal_selective_repair_benchmark.py');cf=load('sh_cf','crossfit_risk_selector_cv.py');eg=load('sh_eg','ensemble_gain_model_cv.py');rows=base.load_real_errors(a.results,a.annotations,a.image_dir);eg.configure_ensemble(base,rows,3);folds=[]
 for outer in range(a.folds):
  _,cr,tr=cf.outer_partitions(rows,outer,a.folds,a.seed);cal,test=cf.as_data(base,cr),cf.as_data(base,tr);methods={};case_rows={}
  methods['no_repair']=row(base,learned,test,0.,'geometry');case_rows['no_repair']=cases(base,test,0.,'geometry')
  for space in ['geometry','output']:
   for label,risk_limit in [('mae_tuned',None),('risk10_tuned',.10)]:
    alpha=choose(base,learned,cal,space,risk_limit);name=f'{space}_{label}';methods[name]=row(base,learned,test,alpha,space);case_rows[name]=cases(base,test,alpha,space)
   for alpha in [.1,.3,.5]:
    name=f'{space}_alpha{str(alpha).replace(".","")}';methods[name]=row(base,learned,test,alpha,space);case_rows[name]=cases(base,test,alpha,space)
  folds.append({'fold':outer,'methods':methods,'cases':case_rows});print(json.dumps({'fold':outer,'methods':methods}),flush=True)
 result={'patient_grouped':True,'calibration_selected_alpha':True,'folds':folds,'aggregate':cf.aggregate(folds)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
