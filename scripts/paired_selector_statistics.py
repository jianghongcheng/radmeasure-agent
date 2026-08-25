#!/usr/bin/env python3
"""Identifier-clustered paired tests for learned selector versus heuristics."""
import argparse,json
from pathlib import Path
import numpy as np

def clustered(data,method):
 groups={}
 for fold in data['folds']:
  for row in fold['cases'][method]:groups.setdefault(row['identifier'],[]).append(row['mean_after'])
 return {key:float(np.mean(value)) for key,value in groups.items()}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/research/selector_baselines_fixed20.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/paired_selector_statistics.json'));p.add_argument('--replicates',type=int,default=20000);a=p.parse_args();data=json.load(open(a.input));learned=clustered(data,'learned_expected_gain');rng=np.random.default_rng(42);result={}
 for method in ['largest_uncertainty','largest_displacement','largest_sensitivity']:
  heuristic=clustered(data,method);keys=sorted(set(learned)&set(heuristic));delta=np.array([heuristic[k]-learned[k] for k in keys]);n=len(delta);boot=delta[rng.integers(0,n,(a.replicates,n))].mean(1);null=(delta*rng.choice([-1,1],(a.replicates,n))).mean(1);observed=delta.mean();result[method]={'n_identifier_clusters':n,'learned_mae_advantage_deg':float(observed),'bootstrap_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'two_sided_sign_flip_p':float((np.abs(null)>=abs(observed)).mean())}
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
