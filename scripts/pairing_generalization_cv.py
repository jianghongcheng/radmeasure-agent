#!/usr/bin/env python3
"""Leave-one-detector-pairing-out generalization with patient-disjoint tests."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

PAIRS=((17,42),(17,73),(42,73))
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def expand_pairs(base,rows,pairs):
 groups={}
 for row in rows:groups.setdefault(row['identifier'],{})[row['seed']]=row
 result=[]
 for identifier,by_seed in groups.items():
  for pair in pairs:
   if not all(seed in by_seed for seed in pair):continue
   for base_seed in pair:
    row=dict(by_seed[base_seed]);stacked=torch.stack([by_seed[seed]['predicted_directions'] for seed in pair]);consensus=base.canonical(stacked.mean(0));deviation=torch.rad2deg(torch.acos((row['predicted_directions']*consensus).sum(-1).abs().clamp(0,1-1e-7)));all_deviation=torch.rad2deg(torch.acos((stacked*consensus[None]).sum(-1).abs().clamp(0,1-1e-7)))
    row['ensemble_consensus_directions']=consensus;row['ensemble_axis_features']=torch.stack([deviation,all_deviation.mean(0)],-1);row['pairing']=f'{pair[0]}-{pair[1]}';result.append(row)
 return result

def subset(data,index):
 index=torch.as_tensor(index,dtype=torch.long);return {key:(value[index] if torch.is_tensor(value) else [value[i] for i in index.tolist()]) for key,value in data.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--coverage',type=float,default=.20)
 p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/pairing_generalization_cv.json'));a=p.parse_args()
 base=load('pg_base','decisive_structured_refinement_benchmark.py');sel=load('pg_sel','selective_axis_verifier_benchmark.py');learned=load('pg_l','learned_proposal_selective_repair_benchmark.py');risk=load('pg_r','risk_adjusted_selector_cv.py');cf=load('pg_cf','crossfit_risk_selector_cv.py');cons=load('pg_c','ensemble_consensus_repair_cv.py');eg=load('pg_eg','ensemble_gain_model_cv.py');sb=load('pg_sb','selector_baselines_cv.py')
 raw=base.load_real_errors(a.results,a.annotations,a.image_dir);cells=[];severity_rows=[]
 for held_pair in PAIRS:
  train_pairs=[pair for pair in PAIRS if pair!=held_pair]
  for outer in range(a.folds):
   rr,_,tr=cf.outer_partitions(raw,outer,a.folds,a.seed);train=cf.as_data(base,expand_pairs(base,rr,train_pairs));test=cf.as_data(base,expand_pairs(base,tr,[held_pair]));train_gain,_=cons.candidates(base,train);gain,candidates=cons.candidates(base,test)
   model=ExtraTreesRegressor(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,random_state=42).fit(eg.features(base,sel,train,'full'),train_gain.reshape(-1).numpy());scores=eg.predict(model,eg.features(base,sel,test,'full'),len(test['targets']));axis,edit=sb.top_budget(scores,a.coverage);pred=cons.choose(base,test,candidates,axis,edit);initial=base.execute(test['directions']);dummy=torch.zeros(len(initial),dtype=torch.long)
   methods={'no_repair':risk.detailed_metrics(base,learned,test,initial,torch.zeros(len(initial),dtype=torch.bool),gain,dummy,.02),'unseen_pairing_expected_gain':risk.detailed_metrics(base,learned,test,pred,edit,gain,axis,.02)}
   selected_gain=gain[torch.arange(len(axis)),axis];rho=float(spearmanr(scores.max(1).values.numpy(),selected_gain.numpy()).statistic);train_error=(base.execute(train['directions'])-train['targets']).abs().mean(1);cuts=torch.quantile(train_error,torch.tensor([1/3,2/3]));test_error=(initial-test['targets']).abs().mean(1);bins=torch.bucketize(test_error,cuts)
   case_rows=[{'identifier':identifier,'mean_before':float(b),'mean_after':float(c)} for identifier,b,c in zip(test['identifiers'],(initial-test['targets']).abs().mean(1),(pred-test['targets']).abs().mean(1))]
   for level,name in enumerate(['low','medium','high']):
    idx=(bins==level).nonzero().flatten()
    if not len(idx):continue
    block=subset(test,idx);block_gain=gain[idx];block_axis=axis[idx];block_edit=edit[idx];block_pred=pred[idx]
    metric=risk.detailed_metrics(base,learned,block,block_pred,block_edit,block_gain,block_axis,.02);bin_rho=float(spearmanr(scores.max(1).values[idx].numpy(),selected_gain[idx].numpy()).statistic) if len(idx)>2 else float('nan');before_mae=float((base.execute(block['directions'])-block['targets']).abs().mean());after_mae=float((block_pred-block['targets']).abs().mean());severity_rows.append({'held_pairing':f'{held_pair[0]}-{held_pair[1]}','fold':outer,'severity':name,'n':len(idx),'score_gain_spearman':bin_rho,'before_MAE':before_mae,'after_MAE':after_mae,'MAE_reduction':before_mae-after_mae,**metric})
   cells.append({'held_pairing':f'{held_pair[0]}-{held_pair[1]}','fold':outer,'score_gain_spearman':rho,'methods':methods,'cases':case_rows});print(json.dumps({key:value for key,value in cells[-1].items() if key!='cases'}),flush=True)
 aggregate=cf.aggregate(cells);weighted={}
 for level in ['low','medium','high']:
  rows=[row for row in severity_rows if row['severity']==level];total=sum(row['n'] for row in rows);weighted[level]={key:sum(row[key]*row['n'] for row in rows)/total for key in ['before_MAE','after_MAE','MAE_reduction','joint_harm_rate','coverage','opportunity_recall','score_gain_spearman']};weighted[level]['n']=total
 result={'leave_one_detector_pairing_out':True,'patient_grouped':True,'coverage_budget':a.coverage,'cells':cells,'aggregate':aggregate,'severity_by_train_terciles':weighted};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'aggregate':aggregate,'severity':weighted},indent=2))
if __name__=='__main__':main()
