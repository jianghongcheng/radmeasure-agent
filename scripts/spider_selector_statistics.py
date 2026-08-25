#!/usr/bin/env python3
"""Pooled and database-clustered statistics for Spider executable edits."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest

def rows(data,method):return [row for fold in data['folds'] for row in fold['methods'][method]['cases']]
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/research/spider_advantage_selector_cv.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/spider_selector_statistics.json'));p.add_argument('--replicates',type=int,default=20000);a=p.parse_args();data=json.load(open(a.input));methods={name:rows(data,name) for name in data['folds'][0]['methods']};pooled={}
 for name,current in methods.items():pooled[name]={'n':len(current),'execution_accuracy':float(np.mean([r['after_correct'] for r in current])),'coverage':float(np.mean([r['edited'] for r in current])),'harm_rate':float(np.mean([r['before_correct'] and not r['after_correct'] for r in current])),'benefit_rate':float(np.mean([not r['before_correct'] and r['after_correct'] for r in current]))}
 learned=methods['learned_exact_advantage'];benefit=sum(not r['before_correct'] and r['after_correct'] for r in learned);harm=sum(r['before_correct'] and not r['after_correct'] for r in learned);dbs=sorted({r['db_id'] for r in learned});by_db={db:[r for r in learned if r['db_id']==db] for db in dbs};rng=np.random.default_rng(42);boot=[]
 for _ in range(a.replicates):
  sampled=rng.choice(dbs,len(dbs),replace=True);selected=[row for db in sampled for row in by_db[db]];boot.append(np.mean([row['after_correct']-row['before_correct'] for row in selected]))
 result={'pooled':pooled,'learned_vs_no_repair':{'beneficial_cases':benefit,'harmed_cases':harm,'net_cases':benefit-harm,'absolute_accuracy_gain':pooled['learned_exact_advantage']['execution_accuracy']-pooled['no_repair']['execution_accuracy'],'database_cluster_bootstrap_95_ci':[float(x) for x in np.quantile(boot,[.025,.975])],'exact_paired_binomial_p':float(binomtest(min(benefit,harm),benefit+harm,.5,alternative='two-sided').pvalue),'n_database_clusters':len(dbs)}};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
