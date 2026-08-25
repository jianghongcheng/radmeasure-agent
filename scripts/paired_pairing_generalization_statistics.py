#!/usr/bin/env python3
"""Clustered uncertainty for leave-one-pairing-out correction."""
import argparse,json
from pathlib import Path
import numpy as np

def statistic(rows,repeats,rng):
 groups={}
 for row in rows:groups.setdefault(row['identifier'],[]).append(row['mean_before']-row['mean_after'])
 delta=np.array([np.mean(value) for value in groups.values()]);n=len(delta);boot=delta[rng.integers(0,n,(repeats,n))].mean(1);null=(delta*rng.choice([-1,1],(repeats,n))).mean(1);observed=delta.mean();return {'n_identifier_clusters':n,'mean_improvement_deg':float(observed),'bootstrap_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'two_sided_sign_flip_p':float((np.abs(null)>=abs(observed)).mean())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/research/pairing_generalization_cv.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/paired_pairing_generalization_statistics.json'));p.add_argument('--replicates',type=int,default=20000);a=p.parse_args();data=json.load(open(a.input));rng=np.random.default_rng(42);all_rows=[row for cell in data['cells'] for row in cell['cases']];result={'all_unseen_pairings':statistic(all_rows,a.replicates,rng)}
 for pairing in sorted({cell['held_pairing'] for cell in data['cells']}):result[pairing]=statistic([row for cell in data['cells'] if cell['held_pairing']==pairing for row in cell['cases']],a.replicates,rng)
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
