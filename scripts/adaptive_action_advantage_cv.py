#!/usr/bin/env python3
"""Exact counterfactual advantage learning over (component, step-size) actions."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

ALPHAS=torch.tensor([.1,.2,.3,.5,.7,1.],dtype=torch.float32)
def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def actions(base,selector,feature_module,data):
 b=len(data['targets']);axis_features=torch.tensor(feature_module.features(base,selector,data,'full')).reshape(b,3,-1);predictions=[];features=[];initial=base.execute(data['directions']);before=(initial-data['targets']).abs().mean(1)
 for axis in range(3):
  axis_predictions=[];axis_action_features=[]
  current=data['directions'][:,axis];proposal=data['ensemble_consensus_directions'][:,axis];delta=proposal-current
  for alpha in ALPHAS:
   directions=data['directions'].clone();blended=torch.nn.functional.normalize(current+alpha*delta,dim=-1);directions[:,axis]=blended;measurement=base.execute(directions);axis_predictions.append(measurement);extra=torch.cat([torch.full((b,1),float(alpha)),torch.full((b,1),float(alpha**2)),alpha*delta,blended,measurement/90],1);axis_action_features.append(torch.cat([axis_features[:,axis],extra],1))
  predictions.append(torch.stack(axis_predictions,1));features.append(torch.stack(axis_action_features,1))
 predictions=torch.stack(predictions,1);features=torch.stack(features,1);after=(predictions-data['targets'][:,None,None,:]).abs().mean(-1);gain=before[:,None,None]-after
 return features,predictions,gain

def fit_models(x,gain):
 flat_x=x.reshape(-1,x.shape[-1]).numpy();flat_gain=gain.reshape(-1).numpy();reg=ExtraTreesRegressor(n_estimators=700,min_samples_leaf=4,max_features=.8,n_jobs=-1,random_state=42).fit(flat_x,flat_gain);benefit=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,class_weight='balanced',random_state=43).fit(flat_x,flat_gain>.5);harm=ExtraTreesClassifier(n_estimators=500,min_samples_leaf=4,max_features=.8,n_jobs=-1,class_weight='balanced',random_state=44).fit(flat_x,flat_gain<-.5);return reg,benefit,harm

def positive_probability(model,x):
 probabilities=model.predict_proba(x);classes=list(model.classes_);return probabilities[:,classes.index(True)] if True in classes else np.zeros(len(x))

def scores(models,x):
 shape=x.shape[:3];flat=x.reshape(-1,x.shape[-1]).numpy();reg,benefit,harm=models;return (torch.tensor(reg.predict(flat),dtype=torch.float32).reshape(shape),torch.tensor(positive_probability(benefit,flat),dtype=torch.float32).reshape(shape),torch.tensor(positive_probability(harm,flat),dtype=torch.float32).reshape(shape))

def execute(data,predictions,score,threshold=0.,allowed_alpha=None):
 b=score.shape[0];flat_score=score.reshape(b,-1)
 if allowed_alpha is not None:
  mask=torch.ones_like(score,dtype=torch.bool);mask[:,:,allowed_alpha]=False;flat_score=score.masked_fill(mask,-float('inf')).reshape(b,-1)
 value,index=flat_score.max(1);edit=value>threshold;flat_predictions=predictions.reshape(b,-1,2);selected=flat_predictions[torch.arange(b),index];initial=EXECUTE(data['directions']);prediction=torch.where(edit[:,None],selected,initial);alpha_index=index%len(ALPHAS);return prediction,edit,index,ALPHAS[alpha_index]

def metrics(data,prediction,edit,alpha):
 before=(EXECUTE(data['directions'])-data['targets']).abs().mean(1);after=(prediction-data['targets']).abs().mean(1);w=after-before;result={'mean_MAE':float(after.mean()),'P90_MAE':float(torch.quantile(after,.9)),'P95_MAE':float(torch.quantile(after,.95)),'coverage':float(edit.float().mean()),'mean_alpha_when_edited':float(alpha[edit].mean()) if edit.any() else 0.,'harm_any':float((w>0).float().mean())}
 for tau in [.5,1.,2.]:result[f'harm_at_{tau:g}deg']=float((w>tau).float().mean())
 result['benefit_at_0.5deg']=float((w<-.5).float().mean());return result

def calibrate(data,predictions,reg,benefit,harm,tau,epsilon,mode):
 rows=[]
 for risk_weight in [0,.25,.5,1,2,4]:
  utility=(reg+.15*benefit-risk_weight*harm) if mode=='hybrid' else (benefit-risk_weight*harm);maximum=utility.reshape(len(utility),-1).max(1).values;thresholds=torch.cat([torch.tensor([0.]),torch.quantile(maximum,torch.linspace(0,1,21))]).unique()
  for threshold in thresholds:
   prediction,edit,_,alpha=execute(data,predictions,utility,float(threshold));row=metrics(data,prediction,edit,alpha);rows.append({'risk_weight':risk_weight,'threshold':float(threshold),**row})
 feasible=[row for row in rows if row[f'harm_at_{tau:g}deg']<=epsilon];return min(feasible,key=lambda row:(row['mean_MAE'],row['harm_any']))

def cases(data,prediction,edit,alpha):
 before=(EXECUTE(data['directions'])-data['targets']).abs().mean(1);after=(prediction-data['targets']).abs().mean(1);return [{'identifier':identifier,'before':float(b),'after':float(c),'edited':bool(e),'alpha':float(a) if e else 0.} for identifier,b,c,e,a in zip(data['identifiers'],before,after,edit,alpha)]

def main():
 global EXECUTE
 p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/adaptive_action_advantage_cv.json'));a=p.parse_args()
 base=load('aa_base','decisive_structured_refinement_benchmark.py');selector=load('aa_sel','selective_axis_verifier_benchmark.py');cf=load('aa_cf','crossfit_risk_selector_cv.py');eg=load('aa_eg','ensemble_gain_model_cv.py');EXECUTE=base.execute;rows=base.load_real_errors(a.results,a.annotations,a.image_dir);eg.configure_ensemble(base,rows,3);folds=[]
 for outer in range(a.folds):
  rr,cr,tr=cf.outer_partitions(rows,outer,a.folds,a.seed);train,cal,test=[cf.as_data(base,x) for x in [rr,cr,tr]];train_x,_,train_gain=actions(base,selector,eg,train);cal_x,cal_predictions,cal_gain=actions(base,selector,eg,cal);test_x,test_predictions,test_gain=actions(base,selector,eg,test);models=fit_models(train_x,train_gain);cal_scores=scores(models,cal_x);test_scores=scores(models,test_x);methods={};case_rows={};zeros=torch.zeros(len(test['targets']),dtype=torch.bool);initial=base.execute(test['directions']);zero_alpha=torch.zeros(len(zeros));methods['no_repair']=metrics(test,initial,zeros,zero_alpha)
  raw_prediction,raw_edit,_,raw_alpha=execute(test,test_predictions,test_scores[0],0.);methods['adaptive_regression']=metrics(test,raw_prediction,raw_edit,raw_alpha);case_rows['adaptive_regression']=cases(test,raw_prediction,raw_edit,raw_alpha)
  binary_prediction,binary_edit,_,binary_alpha=execute(test,test_predictions,test_scores[0],0.,allowed_alpha=len(ALPHAS)-1);methods['binary_regression']=metrics(test,binary_prediction,binary_edit,binary_alpha);case_rows['binary_regression']=cases(test,binary_prediction,binary_edit,binary_alpha)
  for tau,epsilon in [(.5,.02),(.5,.05),(1.,.01)]:
   for mode in ['hybrid','classification']:
    policy=calibrate(cal,cal_predictions,*cal_scores,tau,epsilon,mode);utility=(test_scores[0]+.15*test_scores[1]-policy['risk_weight']*test_scores[2]) if mode=='hybrid' else (test_scores[1]-policy['risk_weight']*test_scores[2]);prediction,edit,_,alpha=execute(test,test_predictions,utility,policy['threshold']);name=f'adaptive_{mode}|tau={tau:g}|epsilon={epsilon:g}';methods[name]={**metrics(test,prediction,edit,alpha),'risk_weight':policy['risk_weight'],'threshold':policy['threshold']};case_rows[name]=cases(test,prediction,edit,alpha)
  flat_gain=test_gain.reshape(len(test_gain),-1);oracle_value,oracle_index=flat_gain.max(1);oracle_edit=oracle_value>0;oracle_prediction=test_predictions.reshape(len(test_gain),-1,2)[torch.arange(len(test_gain)),oracle_index];oracle_prediction=torch.where(oracle_edit[:,None],oracle_prediction,initial);oracle_alpha=ALPHAS[oracle_index%len(ALPHAS)];methods['oracle_adaptive']=metrics(test,oracle_prediction,oracle_edit,oracle_alpha)
  binary_gain=test_gain[:,:,-1];binary_value,binary_axis=binary_gain.max(1);binary_oracle_edit=binary_value>0;binary_oracle_prediction=test_predictions[:,:,-1][torch.arange(len(test_gain)),binary_axis];binary_oracle_prediction=torch.where(binary_oracle_edit[:,None],binary_oracle_prediction,initial);methods['oracle_binary']=metrics(test,binary_oracle_prediction,binary_oracle_edit,torch.ones(len(test_gain)))
  folds.append({'fold':outer,'methods':methods,'cases':case_rows});print(json.dumps({'fold':outer,'methods':methods}),flush=True)
 result={'patient_grouped':True,'exact_counterfactual_labels':True,'actions':{'components':3,'alphas':ALPHAS.tolist(),'stop':True},'folds':folds,'aggregate':cf.aggregate(folds)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
