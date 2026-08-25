#!/usr/bin/env python3
"""Cluster-level power audit without reading sealed SQL labels or outcomes."""
from __future__ import annotations
import argparse,json,math
from collections import Counter,defaultdict
from pathlib import Path
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
def cluster_se(cluster_sum,cluster_n):
 m=len(cluster_sum);total_n=sum(cluster_n.values());effect=sum(cluster_sum.values())/total_n;influence=[cluster_sum[key]-effect*cluster_n[key] for key in cluster_sum];return math.sqrt(m/(m-1)*sum(value*value for value in influence))/total_n,effect
def cluster_effects(quad,method):
 values=defaultdict(list)
 for fold in quad['folds']:
  for case in fold['methods'][method]['cases']:values[case['db_id']].append(int(case['after_correct'])-int(case['before_correct']))
 return [sum(rows)/len(rows) for rows in values.values()]
def sample_sd(values):
 mean=sum(values)/len(values);return math.sqrt(sum((value-mean)**2 for value in values)/(len(values)-1))
def power(effect,sd,n,alpha):
 se=sd/math.sqrt(n);critical=norm.ppf(1-alpha/2);return float(norm.cdf(effect/se-critical)+norm.cdf(-effect/se-critical))
def main():
 p=argparse.ArgumentParser();p.add_argument('--effect',type=float,default=.03);p.add_argument('--alpha',type=float,default=.05);p.add_argument('--power',type=float,default=.8);p.add_argument('--output',type=Path,default=ROOT/'outputs/research/spider_sealed_power_analysis.json');a=p.parse_args();seal=json.loads((ROOT/'outputs/research/spider_protocol_seal_v1.json').read_text());quad=json.loads((ROOT/'outputs/research/spider_clean_base_quadruple_cv.json').read_text());cases=[case for fold in quad['folds'] for case in fold['methods']['learned_selector_learned_proposal']['cases']];cluster_sum=defaultdict(int);cluster_n=Counter()
 for case in cases:cluster_n[case['db_id']]+=1;cluster_sum[case['db_id']]+=int(case['after_correct'])-int(case['before_correct'])
 dev_se,observed=cluster_se(cluster_sum,cluster_n);all_rows=json.loads((ROOT/'third_party/spider_data/spider_data/train_spider.json').read_text())+json.loads((ROOT/'third_party/spider_data/spider_data/train_others.json').read_text());sealed=set(seal['confirmatory_databases']);sealed_counts=Counter(row['db_id'] for row in all_rows if row['db_id'] in sealed);m_dev=len(cluster_n);m_sealed=len(sealed_counts);critical=norm.ppf(1-a.alpha/2);target_z=critical+norm.ppf(a.power);heterogeneity={method:sample_sd(cluster_effects(quad,method)) for method in ['learned_selector_learned_proposal','oracle_selector_learned_proposal','candidate_oracle']};proxy_sd=heterogeneity['candidate_oracle'];required_clusters=math.ceil((target_z*proxy_sd/a.effect)**2);sensitivity={str(sd):{'power':power(a.effect,sd,m_sealed,a.alpha),'required_clusters':math.ceil((target_z*sd/a.effect)**2)} for sd in [.01,.03,.05,.075,.1,.15,.2]};projected_power=power(a.effect,proxy_sd,m_sealed,a.alpha);result={'confirmatory_gold_read':False,'primary_minimum_effect':a.effect,'two_sided_alpha':a.alpha,'target_power':a.power,'development_clusters':m_dev,'development_examples':sum(cluster_n.values()),'observed_dev_effect':observed,'warning_about_observed_policy_variance':'The near-always-STOP policy has artificially low variance and is not used for the primary power decision.','development_between_database_sd':heterogeneity,'power_proxy':'candidate-oracle between-database SD, chosen as the most heterogeneous reachable policy diagnostic','sealed_clusters':m_sealed,'sealed_examples_counted_without_labels':sum(sealed_counts.values()),'normal_approximation_power_at_3pt':projected_power,'estimated_clusters_required_for_target_power':required_clusters,'additional_clusters_needed':max(0,required_clusters-m_sealed),'sd_sensitivity_grid':sensitivity,'assumption':'Candidate-oracle database heterogeneity upper-bounds the eventual learned policy heterogeneity; sensitivity grid shows failure if this assumption is false.','decision':'conditionally_adequate_under_oracle_heterogeneity' if projected_power>=a.power else 'underpowered'};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
