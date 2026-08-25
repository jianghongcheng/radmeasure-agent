#!/usr/bin/env python3
"""Identifier-clustered paired statistics for adaptive action policies."""
import argparse,json
from pathlib import Path
import numpy as np

def values(data,method,field='after'):
 groups={}
 for fold in data['folds']:
  for row in fold['cases'][method]:groups.setdefault(row['identifier'],[]).append(row[field])
 return {key:float(np.mean(value)) for key,value in groups.items()}

def test(left,right,repeats,rng):
 keys=sorted(set(left)&set(right));delta=np.array([right[k]-left[k] for k in keys]);n=len(delta);boot=delta[rng.integers(0,n,(repeats,n))].mean(1);null=(delta*rng.choice([-1,1],(repeats,n))).mean(1);return {'n_identifier_clusters':n,'left_mae_advantage_deg':float(delta.mean()),'bootstrap_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'two_sided_sign_flip_p':float((np.abs(null)>=abs(delta.mean())).mean())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/research/adaptive_action_advantage_cv.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/paired_adaptive_action_statistics.json'));p.add_argument('--replicates',type=int,default=20000);a=p.parse_args();data=json.load(open(a.input));rng=np.random.default_rng(42);adaptive=values(data,'adaptive_regression');binary=values(data,'binary_regression');safe_name='adaptive_hybrid|tau=1|epsilon=0.01';safe=values(data,safe_name);before=values(data,safe_name,'before');result={'adaptive_vs_binary':test(adaptive,binary,a.replicates,rng),'risk_targeted_adaptive_vs_no_repair':test(safe,before,a.replicates,rng)};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
