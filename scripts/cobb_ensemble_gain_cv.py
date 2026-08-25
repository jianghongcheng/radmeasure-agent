#!/usr/bin/env python3
"""Cross-protocol selective correction for AASCE Cobb endplate axes."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor

def canonical(x):
 x=x/x.norm(dim=-1,keepdim=True).clamp_min(1e-7);return x*torch.where(x[...,1:2]<0,-1.,1.)
def execute(d):
 d=canonical(d);c=(d[:,0]*d[:,1]).sum(-1).abs().clamp(0,1-1e-7);return torch.rad2deg(torch.acos(c))
def fold(identifier,k,seed):return int(hashlib.sha1(f'{seed}:{identifier}'.encode()).hexdigest()[:8],16)%k

def load_data(results_path,root):
 runs=json.load(open(results_path)); meta={}
 for name in ['aasce_gt.json','aasce_test_split.json']:
  for row in json.load(open(root/'AASCE'/name)):
   if row['image_name'] not in meta:
    with Image.open(row['image_path']) as im:w,h=im.size
    meta[row['image_name']]=(w,h)
 rows=[]
 for run in runs:
  t=run['test'];targets=t['measurements']['cobb']['ground_truth']
  for identifier,target,item in zip(t['identifiers'],targets,t['predicted_endplate_axes']):
   w,h=meta[identifier];dn=torch.tensor(item['directions_normalized_image'],dtype=torch.float32)
   dp=canonical(dn*torch.tensor([w,h],dtype=torch.float32))
   rows.append({'identifier':identifier,'seed':run['seed'],'directions':dp,'target':float(target),'aspect':np.log(w/h)})
 groups={}
 for r in rows:groups.setdefault(r['identifier'],[]).append(r)
 for group in groups.values():
  stack=torch.stack([r['directions'] for r in group]);cons=canonical(stack.mean(0));dev=torch.rad2deg(torch.acos((stack*cons).sum(-1).abs().clamp(0,1-1e-7)));disp=dev.mean(0)
  for i,r in enumerate(group):r['consensus']=cons;r['ensemble']=torch.stack([dev[i],disp],-1)
 return rows

def tensorize(rows):
 return {'ids':[r['identifier'] for r in rows],'directions':torch.stack([r['directions'] for r in rows]),'consensus':torch.stack([r['consensus'] for r in rows]),'ensemble':torch.stack([r['ensemble'] for r in rows]),'target':torch.tensor([r['target'] for r in rows]),'aspect':torch.tensor([[r['aspect']] for r in rows],dtype=torch.float32)}
def candidate_values(data):
 vals=[]
 for a in range(2):
  d=data['directions'].clone();d[:,a]=data['consensus'][:,a];vals.append(execute(d))
 vals=torch.stack(vals,1);before=(execute(data['directions'])-data['target']).abs()/5;after=(vals-data['target'][:,None]).abs()/5
 return before[:,None]-after,vals
def features(data):
 n=len(data['target']);delta=data['consensus']-data['directions'];angle=(execute(data['directions'])/90)[:,None,None].expand(-1,2,-1);aspect=data['aspect'][:,None].expand(-1,2,-1);one=torch.eye(2)[None].expand(n,-1,-1)
 return torch.cat([data['ensemble'],data['directions'],data['consensus'],delta,angle,aspect,one],-1).reshape(-1,12).numpy()
def choose(data,vals,axis,edit):
 initial=execute(data['directions']);selected=vals[torch.arange(len(axis)),axis];return torch.where(edit,selected,initial)
def metrics(data,pred,edit,gain,axis):
 before=(execute(data['directions'])-data['target']).abs();after=(pred-data['target']).abs();selected=gain[torch.arange(len(axis)),axis];opp=gain.max(1).values>.02
 return {'Cobb_MAE':after.mean().item(),'coverage':edit.float().mean().item(),'harm_rate':(after>before).float().mean().item(),'success_rate':(after<before-.25).float().mean().item(),'mean_normalized_gain':((before-after)/5).mean().item(),'opportunity_prevalence':opp.float().mean().item(),'opportunity_recall':((edit&opp).sum()/opp.sum().clamp_min(1)).item(),'conditional_harm':((edit&(selected<-.02)).sum()/edit.sum().clamp_min(1)).item()}
def calibrate(data,gain,vals,scores,limit):
 scores=scores.float();value,axis=scores.max(1);ts=torch.cat([torch.tensor([value.min()-1e-3,value.max()+1e-3]),torch.quantile(value,torch.linspace(0,1,41))]).unique();out=[]
 for t in ts:
  edit=value>t;out.append({'threshold':float(t),**metrics(data,choose(data,vals,axis,edit),edit,gain,axis)})
 feasible=[r for r in out if r['harm_rate']<=limit];return max(feasible,key=lambda r:(r['mean_normalized_gain'],r['opportunity_recall']))
def aggregate(folds):
 out={}
 for method in folds[0]['methods']:
  out[method]={}
  for metric in folds[0]['methods'][method]:
   x=np.array([f['methods'][method][metric] for f in folds]);out[method][metric]={'mean':float(x.mean()),'sd':float(x.std(ddof=1))}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/aasce_axis_geometry_final/results.json'));p.add_argument('--root',type=Path,default=Path('/media/max/a/caxp (Copy 2)'));p.add_argument('--output',type=Path,default=Path('outputs/research/cobb_ensemble_gain_cv.json'));p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--risk-limit',type=float,default=.10);a=p.parse_args()
 rows=load_data(a.results,a.root);folds=[]
 for testfold in range(a.folds):
  calfold=(testfold+1)%a.folds;train=tensorize([r for r in rows if fold(r['identifier'],a.folds,a.seed) not in {testfold,calfold}]);cal=tensorize([r for r in rows if fold(r['identifier'],a.folds,a.seed)==calfold]);test=tensorize([r for r in rows if fold(r['identifier'],a.folds,a.seed)==testfold]);tg,_=candidate_values(train);cg,cv=candidate_values(cal);eg,ev=candidate_values(test);methods={}
  factories={'extra_trees':ExtraTreesRegressor(n_estimators=500,min_samples_leaf=3,max_features=.8,n_jobs=-1,random_state=42),'hist_gradient_boosting':HistGradientBoostingRegressor(max_iter=200,max_leaf_nodes=12,l2_regularization=1.,random_state=42)}
  for name,model in factories.items():
   model.fit(features(train),tg.reshape(-1).numpy());cs=torch.tensor(model.predict(features(cal))).reshape(-1,2);es=torch.tensor(model.predict(features(test))).reshape(-1,2);policy=calibrate(cal,cg,cv,cs,a.risk_limit);value,axis=es.max(1);edit=value>policy['threshold'];methods[name]=metrics(test,choose(test,ev,axis,edit),edit,eg,axis);methods[name]['threshold']=policy['threshold']
  initial=execute(test['directions']);dummy=eg.argmax(1);methods['no_repair']=metrics(test,initial,torch.zeros(len(dummy),dtype=torch.bool),eg,dummy);oa,og=eg.argmax(1),eg.max(1).values;oe=og>0;methods['oracle']=metrics(test,choose(test,ev,oa,oe),oe,eg,oa)
  folds.append({'fold':testfold,'n':len(test['target']),'methods':methods});a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'folds':folds,'aggregate':aggregate(folds)},indent=2)+'\n');print(json.dumps(folds[-1]),flush=True)
 result={'grouping':'image-level; AASCE has no patient identifiers','folds':folds,'aggregate':aggregate(folds)};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
