#!/usr/bin/env python3
"""Identifier-clustered paired bootstrap and permutation statistics."""
import argparse,json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/research/ensemble_gain_model_cv_risk05.json'));p.add_argument('--method',default='extra_trees');p.add_argument('--output',type=Path,default=Path('outputs/research/paired_cluster_statistics.json'));p.add_argument('--replicates',type=int,default=20000);a=p.parse_args();x=json.load(open(a.input));groups={}
 for fold in x['folds']:
  for row in fold['cases'][a.method]:groups.setdefault(row['identifier'],[]).append(row)
 paired=[]
 for identifier,rows in groups.items():
  before=np.mean([(r['HVA_before']+r['IMA_before'])/2 for r in rows]);after=np.mean([(r['HVA_after']+r['IMA_after'])/2 for r in rows]);paired.append((identifier,before,after))
 delta=np.array([b-c for _,b,c in paired]);rng=np.random.default_rng(42);n=len(delta);boot=delta[rng.integers(0,n,(a.replicates,n))].mean(1);sign=rng.choice([-1,1],(a.replicates,n));null=(delta*sign).mean(1);observed=delta.mean();result={'method':a.method,'cluster':'identifier','n_clusters':n,'mean_improvement_deg':float(observed),'bootstrap_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'two_sided_sign_flip_p':float((np.abs(null)>=abs(observed)).mean()),'improved_cluster_fraction':float((delta>0).mean()),'unchanged_cluster_fraction':float((delta==0).mean())};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
