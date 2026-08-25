#!/usr/bin/env python3
"""Recompute key policies under one pooled, identifier-clustered protocol."""
import argparse,json
from pathlib import Path
import numpy as np

def cluster(rows):
 groups={}
 for row in rows:groups.setdefault(row['identifier'],[]).append(row)
 return [{'identifier':key,'before':float(np.mean([r['before'] for r in values])),'after':float(np.mean([r['after'] for r in values])),'edited':float(np.mean([r['edited'] for r in values]))} for key,values in groups.items()]

def metrics(rows):
 before=np.asarray([r['before'] for r in rows]);after=np.asarray([r['after'] for r in rows]);w=after-before;result={'n':len(rows),'mean_MAE':float(after.mean()),'P90_MAE':float(np.quantile(after,.9)),'P95_MAE':float(np.quantile(after,.95)),'worst_MAE':float(after.max()),'coverage':float(np.mean([r['edited'] for r in rows])),'harm_any':float((w>0).mean())}
 for tau in [.5,1.,2.]:result[f'harm_at_{tau:g}deg']=float((w>tau).mean())
 return result

def convert(rows,prefix='mean_'):
 return [{'identifier':r['identifier'],'before':r[prefix+'before'],'after':r[prefix+'after'],'edited':r['edited']} for r in rows]

def bootstrap(left,right,repeats,rng):
 l={r['identifier']:r['after'] for r in cluster(left)};r={x['identifier']:x['after'] for x in cluster(right)};keys=sorted(set(l)&set(r));delta=np.array([r[k]-l[k] for k in keys]);boot=delta[rng.integers(0,len(delta),(repeats,len(delta)))].mean(1);return {'left_mae_advantage_deg':float(delta.mean()),'bootstrap_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))]}

def metric_intervals(rows,repeats,rng):
 values=np.asarray([row['after'] for row in cluster(rows)]);samples=values[rng.integers(0,len(values),(repeats,len(values)))];return {name:{'estimate':float(function(values)),'bootstrap_95_ci':[float(x) for x in np.quantile(function(samples,axis=1),[.025,.975])]} for name,function in [('mean_MAE',np.mean),('P90_MAE',lambda x,axis=None:np.quantile(x,.9,axis=axis)),('P95_MAE',lambda x,axis=None:np.quantile(x,.95,axis=axis))]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--selector',type=Path,default=Path('outputs/research/selector_baselines_fixed20.json'));p.add_argument('--shrinkage',type=Path,default=Path('outputs/research/shrinkage_baseline_cv.json'));p.add_argument('--adaptive',type=Path,default=Path('outputs/research/adaptive_action_advantage_cv.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/reconciled_policy_metrics.json'));p.add_argument('--replicates',type=int,default=20000);a=p.parse_args();selector=json.load(open(a.selector));shrinkage=json.load(open(a.shrinkage));adaptive=json.load(open(a.adaptive));sources={}
 sources['learned_binary20']=convert([r for f in selector['folds'] for r in f['cases']['learned_expected_gain']]);reference=sources['learned_binary20'];sources['no_repair']=[dict(r,after=r['before'],edited=False) for r in reference]
 sources['fixed_output_alpha03']=convert([r for f in shrinkage['folds'] for r in f['cases']['output_alpha03']]);sources['tuned_output_alpha']=convert([r for f in shrinkage['folds'] for r in f['cases']['output_mae_tuned']]);
 for method in ['adaptive_regression','binary_regression','adaptive_hybrid|tau=1|epsilon=0.01','adaptive_classification|tau=0.5|epsilon=0.02']:
  sources[method]=convert([r for f in adaptive['folds'] for r in f['cases'][method]],prefix='')
 result={'protocol':'pooled predictions followed by identifier clustering; percentiles taken once over 176 identifier-level mean errors','methods':{name:{'row_level':metrics(rows),'identifier_clustered':metrics(cluster(rows))} for name,rows in sources.items()}};rng=np.random.default_rng(42);result['paired_cluster_bootstrap']={'learned_binary20_vs_no_repair':bootstrap(sources['learned_binary20'],sources['no_repair'],a.replicates,rng),'risk_targeted_adaptive_vs_no_repair':bootstrap(sources['adaptive_hybrid|tau=1|epsilon=0.01'],sources['no_repair'],a.replicates,rng),'fixed_output_alpha03_vs_risk_targeted_adaptive':bootstrap(sources['fixed_output_alpha03'],sources['adaptive_hybrid|tau=1|epsilon=0.01'],a.replicates,rng)}
 behavior={}
 for method in ['adaptive_regression','adaptive_hybrid|tau=1|epsilon=0.01','adaptive_classification|tau=0.5|epsilon=0.02']:
  raw=[r for f in adaptive['folds'] for r in f['cases'][method]];edited=[r for r in raw if r['edited']];behavior[method]={'stop_rate':1-len(edited)/len(raw),'alpha_distribution':{str(alpha):sum(abs(r['alpha']-alpha)<1e-5 for r in edited)/len(edited) if edited else 0 for alpha in [.1,.2,.3,.5,.7,1.]}}
 result['action_behavior']=behavior;result['clustered_bootstrap_metric_intervals']={name:metric_intervals(rows,a.replicates,rng) for name,rows in sources.items()};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
