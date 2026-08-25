#!/usr/bin/env python3
"""Magnitude-aware harm and tail analysis for selective and continuous updates."""
import argparse,json
from pathlib import Path
import numpy as np

def cluster(rows):
 groups={}
 for row in rows:groups.setdefault(row['identifier'],[]).append(row)
 return [{'identifier':key,'mean_before':float(np.mean([r['mean_before'] for r in values])),'mean_after':float(np.mean([r['mean_after'] for r in values])),'edited':float(np.mean([r['edited'] for r in values]))} for key,values in groups.items()]

def metrics(rows):
 before=np.array([r['mean_before'] for r in rows]);after=np.array([r['mean_after'] for r in rows]);worsening=after-before;positive=worsening[worsening>0]
 result={'n':len(rows),'mean_MAE':float(after.mean()),'P90_MAE':float(np.quantile(after,.90)),'P95_MAE':float(np.quantile(after,.95)),'worst_MAE':float(after.max()),'trigger_rate':float(np.mean([r['edited'] for r in rows])),'harm_any':float((worsening>0).mean())}
 for tau in [.5,1.,2.]:result[f'harm_at_{tau:g}deg']=float((worsening>tau).mean())
 result['mean_worsening_given_worse']=float(positive.mean()) if len(positive) else 0.;result['P95_worsening_given_worse']=float(np.quantile(positive,.95)) if len(positive) else 0.;result['P95_worsening_all_cases']=float(np.quantile(np.maximum(worsening,0),.95));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--selector',type=Path,default=Path('outputs/research/selector_baselines_fixed20.json'));p.add_argument('--shrinkage',type=Path,default=Path('outputs/research/shrinkage_baseline_cv.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/harm_magnitude_analysis.json'));a=p.parse_args();selector=json.load(open(a.selector));shrinkage=json.load(open(a.shrinkage));sources={}
 for method in ['largest_uncertainty','largest_displacement','largest_sensitivity','learned_expected_gain']:
  sources[method]=[row for fold in selector['folds'] for row in fold['cases'][method]]
 reference=sources['learned_expected_gain'];sources['no_repair']=[dict(row,mean_after=row['mean_before'],edited=False) for row in reference]
 for method in ['geometry_alpha01','geometry_alpha03','geometry_alpha05','geometry_mae_tuned','output_alpha01','output_alpha03','output_alpha05','output_mae_tuned']:
  sources[method]=[row for fold in shrinkage['folds'] for row in fold['cases'][method]]
 result={method:{'row_level':metrics(rows),'identifier_clustered':metrics(cluster(rows))} for method,rows in sources.items()};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
