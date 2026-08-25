#!/usr/bin/env python3
"""Database-clustered bootstrap CIs for the clean-base quadruple."""
from __future__ import annotations
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'outputs/research/spider_clean_base_quadruple_cv.json');p.add_argument('--output',type=Path,default=ROOT/'outputs/research/spider_clean_base_quadruple_statistics.json');p.add_argument('--replicates',type=int,default=20000);p.add_argument('--seed',type=int,default=4401);a=p.parse_args();data=json.loads(a.input.read_text());methods=list(data['folds'][0]['methods']);by_method={}
 for method in methods:
  groups=defaultdict(list)
  for fold in data['folds']:
   for case in fold['methods'][method]['cases']:groups[case['db_id']].append(int(case['after_correct'])-int(case['before_correct']))
  by_method[method]=groups
 dbs=sorted(next(iter(by_method.values())));rng=np.random.default_rng(a.seed);result={}
 for method,groups in by_method.items():
  observed=sum(sum(groups[db]) for db in dbs)/sum(len(groups[db]) for db in dbs);samples=[]
  for _ in range(a.replicates):
   selected=rng.choice(dbs,len(dbs),replace=True);samples.append(sum(sum(groups[db]) for db in selected)/sum(len(groups[db]) for db in selected))
  result[method]={'absolute_gain':observed,'database_clustered_95_ci':[float(np.quantile(samples,.025)),float(np.quantile(samples,.975))],'clusters':len(dbs)}
 decision=result['learned_selector_learned_proposal']['absolute_gain']>=.03 and result['learned_selector_learned_proposal']['database_clustered_95_ci'][0]>0;output={'replicates':a.replicates,'seed':a.seed,'methods':result,'preregistered_go':decision,'go_requirements':{'minimum_gain':.03,'ci_lower_gt_zero':True}};a.output.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
if __name__=='__main__':main()
