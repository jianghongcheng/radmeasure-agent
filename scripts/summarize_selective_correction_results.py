#!/usr/bin/env python3
"""Create the frozen selective-correction experiment summary."""
import csv, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs/research'
def read(name): return json.loads((OUT/name).read_text())
def mean(block,method,metric): return block['aggregate'][method][metric]['mean']

hv05=read('ensemble_gain_model_cv_risk05.json');hv10=read('ensemble_gain_model_cv.json');hv15=read('ensemble_gain_model_cv_risk15.json');hv20=read('ensemble_gain_model_cv_risk20.json')
cobb=read('cobb_ensemble_gain_cv.json');disc=read('cobb_discrete_structure_repair_cv.json');failed=read('learned_proposal_cv.json')
fixed15=read('fixed_coverage_15.json');fixed20=read('fixed_coverage_20.json')
fixed20_seeds={2027:fixed20,2028:read('fixed_coverage_20_seed2028.json'),2029:read('fixed_coverage_20_seed2029.json')}
ablations={name:read('ablation_'+name+'.json') for name in ['no_uncertainty','no_sensitivity','geometry_only','uncertainty_only']}
paired05=read('paired_cluster_statistics.json');paired15=read('paired_cluster_statistics_fixed15.json');paired20=read('paired_cluster_statistics_fixed20.json')
selector_baselines=read('selector_baselines_fixed20.json');selector_stats=read('paired_selector_statistics.json');ensemble_cost=read('ensemble_inference_cost.json')
ensemble_sizes={'1':read('ensemble_size_1_fixed20.json'),'2_offset1':read('ensemble_size_2_fixed20.json'),'2_offset2':read('ensemble_size_2_offset2_fixed20.json'),'3':read('ensemble_size_3_fixed20.json')}
pairing_generalization=read('pairing_generalization_cv.json');pairing_statistics=read('paired_pairing_generalization_statistics.json')
shrinkage=read('shrinkage_baseline_cv.json');tails=read('selector_tail_metrics.json')
harm_magnitude=read('harm_magnitude_analysis.json');pareto=read('pareto_frontier_cv.json');calibrated_pareto=read('calibrated_pareto_policies_cv.json')
adaptive_actions=read('adaptive_action_advantage_cv.json');adaptive_statistics=read('paired_adaptive_action_statistics.json')
oracle_diagnostics=read('adaptive_oracle_diagnostics.json');reconciled=read('reconciled_policy_metrics.json')
ltt_feasibility=read('ltt_risk_control_feasibility.json')
spider_edits=read('spider_executable_edits.json');spider_selector=read('spider_advantage_selector_cv.json');spider_statistics=read('spider_selector_statistics.json')
spider_leakage_audit=read('spider_gold_leakage_audit.json')
selector_seed_statistics={'2027':selector_stats,'2028':read('paired_selector_statistics_seed2028.json'),'2029':read('paired_selector_statistics_seed2029.json')}
rows=[]
def add(task,method,source,mae,harm,coverage,opportunity):rows.append(dict(task=task,method=method,source=source,mae=mae,harm=harm,coverage=coverage,opportunity_recall=opportunity))
add('HVA/IMA','No repair','five-fold patient-grouped',mean(hv10,'no_repair','mean_MAE'),0,0,0)
add('HVA/IMA','Unconditional learned repair','five-fold patient-grouped',mean(failed,'learned_repair_all','mean_MAE'),mean(failed,'learned_repair_all','joint_harm_rate'),1,1)
for label,data,method in [('Conservative ensemble gain',hv05,'extra_trees'),('Balanced ensemble gain',hv10,'extra_trees'),('Effect-oriented ensemble gain',hv15,'hist_gradient_boosting'),('Higher-coverage ensemble gain',hv20,'extra_trees')]:
 add('HVA/IMA',label,'five-fold patient-grouped',mean(data,method,'mean_MAE'),mean(data,method,'joint_harm_rate'),mean(data,method,'coverage'),mean(data,method,'opportunity_recall'))
for label,data in [('Fixed 15% intervention budget',fixed15),('Fixed 20% intervention budget',fixed20)]:
 add('HVA/IMA',label,'five-fold patient-grouped',mean(data,'extra_trees','mean_MAE'),mean(data,'extra_trees','joint_harm_rate'),mean(data,'extra_trees','coverage'),mean(data,'extra_trees','opportunity_recall'))
add('HVA/IMA','Oracle selector + consensus proposal','five-fold patient-grouped',mean(hv10,'oracle','mean_MAE'),0,mean(hv10,'oracle','coverage'),1)
add('Cobb axes','No repair','five-fold image-grouped',mean(cobb,'no_repair','Cobb_MAE'),0,0,0)
add('Cobb axes','Selective consensus-axis repair','five-fold image-grouped',mean(cobb,'extra_trees','Cobb_MAE'),mean(cobb,'extra_trees','harm_rate'),mean(cobb,'extra_trees','coverage'),mean(cobb,'extra_trees','opportunity_recall'))
add('Cobb structure','No repair','five-fold image-grouped',mean(disc,'no_repair','Cobb_MAE'),0,0,0)
add('Cobb structure','Selective RESELECT_STRUCTURE','five-fold image-grouped',mean(disc,'extra_trees','Cobb_MAE'),mean(disc,'extra_trees','harm_rate'),mean(disc,'extra_trees','coverage'),mean(disc,'extra_trees','opportunity_recall'))
ablation_rows={name:{metric:mean(data,'extra_trees',metric) for metric in ['mean_MAE','joint_harm_rate','coverage','opportunity_recall']} for name,data in ablations.items()}
split_robustness={str(seed):{'mean_mae_reduction':sum(f['methods']['no_repair']['mean_MAE']-f['methods']['extra_trees']['mean_MAE'] for f in data['folds'])/len(data['folds']),'mean_harm':mean(data,'extra_trees','joint_harm_rate'),'mean_coverage':mean(data,'extra_trees','coverage'),'improved_folds':sum(f['methods']['extra_trees']['mean_MAE']<f['methods']['no_repair']['mean_MAE'] for f in data['folds'])} for seed,data in fixed20_seeds.items()}
summary={'frozen_at':'2026-08-24','rows':rows,'feature_ablation':ablation_rows,'split_seed_robustness_fixed20':split_robustness,
 'selector_baselines_fixed20':{method:{metric:mean(selector_baselines,method,metric) for metric in ['mean_MAE','joint_harm_rate','coverage','opportunity_recall']} for method in selector_baselines['aggregate']},
 'paired_selector_statistics':selector_stats,
 'paired_selector_statistics_by_split_seed':selector_seed_statistics,
 'shrinkage_baselines':shrinkage['aggregate'],'selector_tail_metrics':tails,
 'harm_magnitude_analysis':harm_magnitude,'pareto_frontier_artifact':'pareto_frontier_cv.json','calibrated_pareto_policies':calibrated_pareto['aggregate'],
 'adaptive_action_advantage':adaptive_actions['aggregate'],'paired_adaptive_action_statistics':adaptive_statistics,
 'adaptive_oracle_diagnostics':oracle_diagnostics,'reconciled_policy_metrics':reconciled,
 'ltt_risk_control_feasibility':ltt_feasibility,
 'spider_executable_edit_summary':spider_edits['summary'],'spider_advantage_selector':spider_selector['aggregate'],'spider_selector_statistics':spider_statistics,'spider_gold_leakage_audit':spider_leakage_audit,
 'ensemble_size_fixed20':{size:{'mean_MAE':mean(data,'extra_trees','mean_MAE'),'harm':mean(data,'extra_trees','joint_harm_rate'),'opportunity_recall':mean(data,'extra_trees','opportunity_recall'),'oracle_MAE':mean(data,'oracle','mean_MAE')} for size,data in ensemble_sizes.items()},
 'ensemble_inference_cost':ensemble_cost,
 'leave_one_pairing_out':{'aggregate':pairing_generalization['aggregate'],'severity_by_train_terciles':pairing_generalization['severity_by_train_terciles'],'paired_statistics':pairing_statistics},
 'paired_identifier_statistics':{'risk_constrained_5pct':paired05,'fixed_15pct':paired15,'fixed_20pct':paired20},
 'limitations':['HVAngleEst uses patient grouping; AASCE lacks patient identifiers and uses image grouping.','Cobb continuous-axis and discrete-pair experiments use different upstream models and are separate evaluations.','Reported harm is overall intervention harm; conditional harm is available in source artifacts.','Fixed-budget policies require a deployment batch or rolling score-ranking window.','External institutional validation remains required for clinical generalization.']}
(OUT/'iclr_selective_correction_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'iclr_selective_correction_summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps(summary,indent=2))
