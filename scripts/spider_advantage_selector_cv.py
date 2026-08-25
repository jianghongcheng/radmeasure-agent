#!/usr/bin/env python3
"""Cross-database exact-advantage selection for executable Spider SQL edits."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

def fold_of(db_id,folds,seed):return int(hashlib.sha1(f'{seed}:{db_id}'.encode()).hexdigest()[:8],16)%folds
def text(record,candidate):return f"question {record['question']} baseline {record['predicted_sql']} action {candidate['action']} candidate {candidate['sql']} executable {int(candidate['executable'])}"
def flatten(records):return [(record,candidate,text(record,candidate)) for record in records for candidate in record['candidates']]
def probability(model,x):
 classes=list(model.classes_);return model.predict_proba(x)[:,classes.index(1)] if 1 in classes else np.zeros(x.shape[0])
def train(records):
 rows=flatten(records);vectorizer=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=2,max_features=30000,sublinear_tf=True);x=vectorizer.fit_transform([row[2] for row in rows]);benefit=np.asarray([row[1]['advantage']>0 for row in rows],dtype=int);harm=np.asarray([row[1]['advantage']<0 for row in rows],dtype=int);kwargs=dict(loss='log',class_weight='balanced',max_iter=2000,tol=1e-4,random_state=42);return vectorizer,SGDClassifier(alpha=1e-5,**kwargs).fit(x,benefit),SGDClassifier(alpha=1e-5,**kwargs).fit(x,harm)
def score(records,models):
 vectorizer,benefit,harm=models;rows=flatten(records);x=vectorizer.transform([row[2] for row in rows]);pb,ph=probability(benefit,x),probability(harm,x);grouped={};offset=0
 for record in records:
  n=len(record['candidates']);grouped[record['index']]={'benefit':pb[offset:offset+n],'harm':ph[offset:offset+n]};offset+=n
 return grouped
def evaluate(records,scores,risk_weight,threshold,mode):
 correct=[];harm=[];benefit=[];edited=[];cases=[]
 for record in records:
  if mode=='no_repair' or not record['candidates']:chosen=None
  elif mode=='first_executable':chosen=next((i for i,c in enumerate(record['candidates']) if c['executable']),None)
  elif mode=='oracle':
   chosen=next((i for i,c in enumerate(record['candidates']) if c['correct'] and not record['base_correct']),None)
  else:
   utility=scores[record['index']]['benefit']-risk_weight*scores[record['index']]['harm'];index=int(np.argmax(utility));chosen=index if utility[index]>threshold else None
  after=record['base_correct'] if chosen is None else record['candidates'][chosen]['correct'];is_harm=record['base_correct'] and not after;is_benefit=not record['base_correct'] and after;correct.append(after);harm.append(is_harm);benefit.append(is_benefit);edited.append(chosen is not None);cases.append({'index':record['index'],'db_id':record['db_id'],'before_correct':record['base_correct'],'after_correct':after,'edited':chosen is not None})
 return {'execution_accuracy':float(np.mean(correct)),'harm_rate':float(np.mean(harm)),'benefit_rate':float(np.mean(benefit)),'net_benefit_minus_harm':float(np.mean(benefit)-np.mean(harm)),'coverage':float(np.mean(edited)),'cases':cases}
def calibrate(records,scores):
 maximum=[]
 for record in records:
  if record['candidates']:maximum.extend((scores[record['index']]['benefit']-scores[record['index']]['harm']).tolist())
 thresholds=np.unique(np.r_[0,np.quantile(maximum,np.linspace(0,1,41))]);rows=[]
 for risk_weight in [0,.25,.5,1,2,4,8]:
  for threshold in thresholds:
   metric=evaluate(records,scores,risk_weight,float(threshold),'learned');rows.append({'risk_weight':risk_weight,'threshold':float(threshold),**{k:v for k,v in metric.items() if k!='cases'}})
 return max(rows,key=lambda row:(row['execution_accuracy'],-row['harm_rate']))
def aggregate(folds):
 methods=folds[0]['methods'];return {method:{metric:{'mean':float(np.mean([fold['methods'][method][metric] for fold in folds])),'sd':float(np.std([fold['methods'][method][metric] for fold in folds],ddof=1))} for metric in methods[method] if metric!='cases'} for method in methods}
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/research/spider_executable_edits.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/spider_advantage_selector_cv.json'));p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);a=p.parse_args();records=json.load(open(a.input))['records'];folds=[]
 for outer in range(a.folds):
  test=[r for r in records if fold_of(r['db_id'],a.folds,a.seed)==outer];cal=[r for r in records if fold_of(r['db_id'],a.folds,a.seed)==(outer+1)%a.folds];train_rows=[r for r in records if fold_of(r['db_id'],a.folds,a.seed) not in {outer,(outer+1)%a.folds}];models=train(train_rows);cal_scores=score(cal,models);test_scores=score(test,models);policy=calibrate(cal,cal_scores);methods={}
  for name,mode in [('no_repair','no_repair'),('first_executable_edit','first_executable'),('learned_exact_advantage','learned'),('oracle_candidate','oracle')]:
   metric=evaluate(test,test_scores,policy['risk_weight'],policy['threshold'],mode);methods[name]=metric
  folds.append({'fold':outer,'n':{'train':len(train_rows),'calibration':len(cal),'test':len(test)},'databases':{'test':sorted({r['db_id'] for r in test})},'policy':policy,'methods':methods});print(json.dumps({'fold':outer,'n':folds[-1]['n'],'policy':policy,'methods':{k:{m:v for m,v in x.items() if m!='cases'} for k,x in methods.items()}}),flush=True)
 result={'domain':'Spider 1.0 Text-to-SQL','database_grouped_cv':True,'exact_reward':'SQLite execution-result equality','folds':folds,'aggregate':aggregate(folds)};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['aggregate'],indent=2))
if __name__=='__main__':main()
