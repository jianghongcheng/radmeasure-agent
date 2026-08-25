#!/usr/bin/env python3
"""Proposal/selector quadruple on the clean cross-fitted Spider base."""
from __future__ import annotations
import argparse,hashlib,importlib.util,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def load_selector():
 path=ROOT/'scripts/spider_advantage_selector_cv.py';spec=importlib.util.spec_from_file_location('selector',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def fold_of(db,folds,seed):return int(hashlib.sha256(f'quadruple:{seed}:{db}'.encode()).hexdigest()[:8],16)%folds
def selected_proposals(records,scores):
 output={}
 for record in records:
  if not record['candidates']:output[record['index']]=None;continue
  output[record['index']]=int(np.argmax(scores[record['index']]['benefit']))
 return output
def evaluate(records,scores,proposals,risk_weight,threshold,mode):
 cases=[]
 for record in records:
  index=proposals[record['index']];chosen=None
  if index is not None:
   candidate=record['candidates'][index];utility=float(scores[record['index']]['benefit'][index]-risk_weight*scores[record['index']]['harm'][index])
   if mode=='proposal_all':chosen=index
   elif mode=='learned_selective' and utility>threshold:chosen=index
   elif mode=='oracle_selector' and candidate['advantage']>0:chosen=index
   elif mode=='candidate_oracle':chosen=next((i for i,row in enumerate(record['candidates']) if row['advantage']>0),None)
  after=record['base_correct'] if chosen is None else record['candidates'][chosen]['correct'];cases.append({'index':record['index'],'db_id':record['db_id'],'before_correct':record['base_correct'],'after_correct':after,'edited':chosen is not None})
 return metric(cases)
def metric(cases):
 n=len(cases);benefit=sum(not c['before_correct'] and c['after_correct'] for c in cases);harm=sum(c['before_correct'] and not c['after_correct'] for c in cases)
 return {'n':n,'execution_accuracy':sum(c['after_correct'] for c in cases)/n,'coverage':sum(c['edited'] for c in cases)/n,'benefit_count':benefit,'harm_count':harm,'absolute_gain':(benefit-harm)/n,'cases':cases}
def calibrate(records,scores,proposals):
 rows=[]
 for risk_weight in [0,.25,.5,1,2,4,8]:
  values=[]
  for record in records:
   index=proposals[record['index']]
   if index is not None:values.append(float(scores[record['index']]['benefit'][index]-risk_weight*scores[record['index']]['harm'][index]))
  for threshold in np.unique(np.r_[-np.inf,0,np.quantile(values,np.linspace(0,1,41)),np.inf]):
   result=evaluate(records,scores,proposals,risk_weight,float(threshold),'learned_selective');rows.append({'risk_weight':risk_weight,'threshold':float(threshold),'execution_accuracy':result['execution_accuracy'],'harm_count':result['harm_count'],'coverage':result['coverage']})
 return max(rows,key=lambda row:(row['execution_accuracy'],-row['harm_count'],-row['coverage']))
def pool(folds,method):
 cases=sorted([c for fold in folds for c in fold['methods'][method]['cases']],key=lambda row:row['index']);return metric(cases)
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'outputs/research/spider_clean_base_executable_edits.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/research/spider_clean_base_quadruple_cv.json');p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=9917);a=p.parse_args();records=json.loads(a.input.read_text())['records'];selector=load_selector();folds=[]
 for outer in range(a.folds):
  test=[r for r in records if fold_of(r['db_id'],a.folds,a.seed)==outer];cal=[r for r in records if fold_of(r['db_id'],a.folds,a.seed)==(outer+1)%a.folds];train=[r for r in records if fold_of(r['db_id'],a.folds,a.seed) not in {outer,(outer+1)%a.folds}];models=selector.train(train);cal_scores=selector.score(cal,models);test_scores=selector.score(test,models);cal_proposals=selected_proposals(cal,cal_scores);test_proposals=selected_proposals(test,test_scores);policy=calibrate(cal,cal_scores,cal_proposals);no_cases=[{'index':r['index'],'db_id':r['db_id'],'before_correct':r['base_correct'],'after_correct':r['base_correct'],'edited':False} for r in test];methods={'no_repair':metric(no_cases)}
  for name,mode in [('learned_proposal_all','proposal_all'),('learned_selector_learned_proposal','learned_selective'),('oracle_selector_learned_proposal','oracle_selector'),('candidate_oracle','candidate_oracle')]:methods[name]=evaluate(test,test_scores,test_proposals,policy['risk_weight'],policy['threshold'],mode)
  folds.append({'fold':outer,'databases':{'train':sorted({r['db_id'] for r in train}),'calibration':sorted({r['db_id'] for r in cal}),'test':sorted({r['db_id'] for r in test})},'n':{'train':len(train),'calibration':len(cal),'test':len(test)},'policy':policy,'methods':methods});print(json.dumps({'fold':outer,'n':folds[-1]['n'],'policy':policy,'methods':{k:{x:y for x,y in v.items() if x!='cases'} for k,v in methods.items()}}),flush=True)
 pooled={method:{k:v for k,v in pool(folds,method).items() if k!='cases'} for method in folds[0]['methods']};proposal_headroom=pooled['oracle_selector_learned_proposal']['absolute_gain'];candidate_headroom=pooled['candidate_oracle']['absolute_gain'];selector_gain=pooled['learned_selector_learned_proposal']['absolute_gain'];result={'status':'development-only clean-base quadruple','database_grouped_nested_cv':True,'folds':folds,'pooled':pooled,'decomposition':{'candidate_oracle_headroom':candidate_headroom,'learned_proposal_headroom':proposal_headroom,'proposal_recovery':proposal_headroom/candidate_headroom if candidate_headroom else None,'selector_recovery':selector_gain/proposal_headroom if proposal_headroom else None,'end_to_end_recovery':selector_gain/candidate_headroom if candidate_headroom else None}};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'pooled':pooled,'decomposition':result['decomposition']},indent=2))
if __name__=='__main__':main()
