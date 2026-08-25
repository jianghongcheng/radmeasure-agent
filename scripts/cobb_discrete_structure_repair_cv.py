#!/usr/bin/env python3
"""Selective RESELECT_STRUCTURE for Cobb candidate-pair predictions."""
from __future__ import annotations
import argparse, glob, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting  # noqa:F401
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

def fold(identifier,k,seed):return int(hashlib.sha1(f'{seed}:{identifier}'.encode()).hexdigest()[:8],16)%k
def load_rows(pattern):
 frames=[]
 for i,path in enumerate(sorted(glob.glob(pattern))):
  d=pd.read_csv(path);d=d[d.split=='test'].copy();d['run']=i;frames.append(d)
 allrows=pd.concat(frames,ignore_index=True);out=[]
 for image,group in allrows.groupby('image_path'):
  counts=group.selected_candidate_rank.value_counts();rank=int(counts.index[0]);vote=float(counts.iloc[0]/len(group));matches=group[group.selected_candidate_rank==rank];angle=float(matches.selector_hard_angle.mean())
  for row in group.itertuples():
   gain=(abs(row.selector_hard_angle-row.cobb_gt)-abs(angle-row.cobb_gt))/5
   out.append({'id':image,'gain':gain,'before':row.selector_hard_angle,'proposal':angle,'target':row.cobb_gt,
    'changed':bool(row.selected_candidate_rank!=rank),
    'features':[vote,float(row.selected_candidate_rank==rank),abs(row.selected_candidate_rank-rank)/200,
     row.selection_entropy,row.selection_margin,row.prob_max,row.logit_margin,row.angle_std,row.expected_abs_dev,
     row.top2_angle_gap,row.top5_angle_range,row.soft_hard_gap,row.selector_hard_angle/90,angle/90,(angle-row.selector_hard_angle)/90]})
 return out
def metrics(rows,pred,edit):
 before=np.abs(np.array([r['before'] for r in rows])-np.array([r['target'] for r in rows]));after=np.abs(pred-np.array([r['target'] for r in rows]));gain=np.array([r['gain'] for r in rows]);opp=gain>.02
 return {'Cobb_MAE':float(after.mean()),'coverage':float(edit.mean()),'harm_rate':float((after>before).mean()),'success_rate':float((after<before-.25).mean()),'mean_normalized_gain':float(((before-after)/5).mean()),'opportunity_prevalence':float(opp.mean()),'opportunity_recall':float((edit&opp).sum()/max(opp.sum(),1)),'conditional_harm':float((edit&(gain<-.02)).sum()/max(edit.sum(),1))}
def calibrate(rows,scores,limit):
 scores=np.asarray(scores);qs=np.unique(np.r_[scores.min()-1e-3,scores.max()+1e-3,np.quantile(scores,np.linspace(0,1,41))]);res=[]
 for t in qs:
  edit=(scores>t)&np.array([r['changed'] for r in rows]);pred=np.where(edit,[r['proposal'] for r in rows],[r['before'] for r in rows]);res.append({'threshold':float(t),**metrics(rows,pred,edit)})
 feasible=[r for r in res if r['harm_rate']<=limit];return max(feasible,key=lambda r:(r['mean_normalized_gain'],r['opportunity_recall']))
def aggregate(folds):
 out={}
 for m in folds[0]['methods']:
  out[m]={}
  for key in folds[0]['methods'][m]:
   x=np.array([f['methods'][m][key] for f in folds]);out[m][key]={'mean':float(x.mean()),'sd':float(x.std(ddof=1))}
 return out
def main():
 p=argparse.ArgumentParser();p.add_argument('--pattern',default='/media/max/a/caxp (Copy 2)/cvpr/outputs/stage2_pair_selector*/stage2_pair_selector_predictions.csv');p.add_argument('--output',type=Path,default=Path('outputs/research/cobb_discrete_structure_repair_cv.json'));p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--risk-limit',type=float,default=.10);a=p.parse_args();rows=load_rows(a.pattern);folds=[]
 for tf in range(a.folds):
  cf=(tf+1)%a.folds;train=[r for r in rows if fold(r['id'],a.folds,a.seed) not in {tf,cf}];cal=[r for r in rows if fold(r['id'],a.folds,a.seed)==cf];test=[r for r in rows if fold(r['id'],a.folds,a.seed)==tf];x=np.array([r['features'] for r in train]);y=np.array([r['gain'] for r in train]);methods={}
  models={'extra_trees':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,random_state=42),'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=15,l2_regularization=1.,random_state=42)}
  for name,model in models.items():
   model.fit(x,y);policy=calibrate(cal,model.predict([r['features'] for r in cal]),a.risk_limit);score=model.predict([r['features'] for r in test]);edit=(score>policy['threshold'])&np.array([r['changed'] for r in test]);pred=np.where(edit,[r['proposal'] for r in test],[r['before'] for r in test]);methods[name]={**metrics(test,pred,edit),'threshold':policy['threshold']}
  methods['no_repair']=metrics(test,np.array([r['before'] for r in test]),np.zeros(len(test),bool));gain=np.array([r['gain'] for r in test]);oe=gain>0;methods['oracle']=metrics(test,np.where(oe,[r['proposal'] for r in test],[r['before'] for r in test]),oe)
  folds.append({'fold':tf,'n':len(test),'methods':methods});a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'folds':folds,'aggregate':aggregate(folds)},indent=2)+'\n');print(json.dumps(folds[-1]),flush=True)
 result={'action':'RESELECT_STRUCTURE to ensemble-majority candidate pair','grouping':'image-level','folds':folds,'aggregate':aggregate(folds)};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
