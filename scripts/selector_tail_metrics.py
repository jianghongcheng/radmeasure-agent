#!/usr/bin/env python3
"""Tail and asymmetric outcome metrics for matched selectors."""
import argparse,json
from pathlib import Path
import numpy as np

def summarize(rows,delta=.02):
 before=np.asarray([row['mean_before'] for row in rows]);after=np.asarray([row['mean_after'] for row in rows]);change=before-after;edited=np.asarray([row['edited'] for row in rows],dtype=float);worsening=np.maximum(-change,0)
 return {'n_policy_outputs':len(rows),'mean_MAE':float(after.mean()),'P90_MAE':float(np.quantile(after,.9)),'P95_MAE':float(np.quantile(after,.95)),'worst_MAE':float(after.max()),'trigger_rate':float(edited.mean()),'benefit_rate':float((change>delta).mean()),'harm_rate':float((change<-delta).mean()),'net_benefit_minus_harm':float((change>delta).mean()-(change<-delta).mean()),'P95_harm_magnitude':float(np.quantile(worsening,.95)),'mean_harm_magnitude_among_harmed':float(worsening[worsening>delta].mean()) if (worsening>delta).any() else 0.}

def cluster(rows):
 groups={}
 for row in rows:groups.setdefault(row['identifier'],[]).append(row)
 return [{'identifier':identifier,'mean_before':float(np.mean([r['mean_before'] for r in values])),'mean_after':float(np.mean([r['mean_after'] for r in values])),'edited':float(np.mean([r['edited'] for r in values]))} for identifier,values in groups.items()]

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/research/selector_baselines_fixed20.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/selector_tail_metrics.json'));a=p.parse_args();data=json.load(open(a.input));methods=['random','largest_uncertainty','largest_displacement','largest_sensitivity','learned_expected_gain'];result={}
 for method in methods:
  rows=[row for fold in data['folds'] for row in fold['cases'][method]];result[method]={'row_level':summarize(rows),'identifier_clustered':summarize(cluster(rows))}
 reference=[row for fold in data['folds'] for row in fold['cases']['learned_expected_gain']];rows=[dict(row,mean_after=row['mean_before'],edited=False) for row in reference];result['no_repair']={'row_level':summarize(rows),'identifier_clustered':summarize(cluster(rows))};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
