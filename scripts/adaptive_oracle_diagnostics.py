#!/usr/bin/env python3
"""Diagnose adaptive-oracle headroom, nesting, and max-over-actions bias."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np, torch

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def summarize(initial,gain):
 selected=gain.clamp_min(0);after=initial-selected;return {'mean_MAE':float(after.mean()),'mean_improvement':float(selected.mean()),'P90_MAE':float(torch.quantile(after,.9)),'P95_MAE':float(torch.quantile(after,.95)),'intervention_rate':float((gain>0).float().mean())}

def main():
 p=argparse.ArgumentParser();p.add_argument('--replicates',type=int,default=2000);p.add_argument('--seed',type=int,default=42);p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/adaptive_oracle_diagnostics.json'));a=p.parse_args()
 base=load('od_base','decisive_structured_refinement_benchmark.py');selector=load('od_sel','selective_axis_verifier_benchmark.py');eg=load('od_eg','ensemble_gain_model_cv.py');adaptive=load('od_ad','adaptive_action_advantage_cv.py');rows=base.load_real_errors(a.results,a.annotations,a.image_dir);eg.configure_ensemble(base,rows,3);data=base.tensors([dict(row,split='all') for row in rows],'all');_,_,gain=adaptive.actions(base,selector,eg,data);initial=(base.execute(data['directions'])-data['targets']).abs().mean(1);result={'n_rows':len(initial),'n_identifiers':len(set(data['identifiers'])),'binary_action_is_subset':True,'alphas':adaptive.ALPHAS.tolist(),'fixed_component_alpha1':{},'fixed_component_adaptive_alpha':{}}
 for component in range(3):result['fixed_component_alpha1'][str(component)]=summarize(initial,gain[:,component,-1]);result['fixed_component_adaptive_alpha'][str(component)]=summarize(initial,gain[:,component].max(1).values)
 binary_gain=gain[:,:,-1].max(1).values;adaptive_gain=gain.reshape(len(gain),-1).max(1).values;result['all_components_alpha1']=summarize(initial,binary_gain);result['all_components_per_case_alpha']=summarize(initial,adaptive_gain)
 global_rows=[]
 for alpha_index,alpha in enumerate(adaptive.ALPHAS):global_rows.append({'alpha':float(alpha),'summary':summarize(initial,gain[:,:,alpha_index].max(1).values)})
 result['global_alpha_oracle']=min(global_rows,key=lambda row:row['summary']['mean_MAE']);result['global_alpha_grid']=global_rows
 groups={}
 for index,identifier in enumerate(data['identifiers']):groups.setdefault(identifier,[]).append(index)
 blocks=[np.asarray(indices) for indices in groups.values()];assert len({len(block) for block in blocks})==1;block_array=np.stack(blocks);rng=np.random.default_rng(a.seed);null_gap=[];random_alpha_gap=[]
 gain_np=gain.numpy()
 for _ in range(a.replicates):
  permuted=np.empty_like(gain_np)
  for component in range(3):
   for alpha_index in range(len(adaptive.ALPHAS)):
    order=rng.permutation(len(block_array));permuted[block_array.reshape(-1),component,alpha_index]=gain_np[block_array[order].reshape(-1),component,alpha_index]
  blocked=permuted[block_array];null_binary=np.maximum(blocked[:,:,:, -1].max(2),0);null_adaptive=np.maximum(blocked.reshape(len(block_array),block_array.shape[1],-1).max(2),0);null_gap.append(float((null_adaptive-null_binary).mean()))
  assigned=rng.integers(0,len(adaptive.ALPHAS),len(block_array));chosen=np.stack([gain_np[block_array[i],:,assigned[i]] for i in range(len(block_array))]);random_gain=np.maximum(chosen.max(2),0);observed_binary=np.maximum(gain_np[block_array][:,:,:,-1].max(2),0);random_alpha_gap.append(float((random_gain-observed_binary).mean()))
 observed_gap=float((adaptive_gain.clamp_min(0)-binary_gain.clamp_min(0)).mean());null=np.asarray(null_gap);random_gap=np.asarray(random_alpha_gap);result['adaptive_minus_binary_observed_deg']=observed_gap;result['block_permutation_null']={'definition':'independently permute identifier blocks for every component-alpha action, preserving each action marginal while destroying case-action correspondence','mean_gap_deg':float(null.mean()),'95_interval':[float(np.quantile(null,.025)),float(np.quantile(null,.975))],'observed_minus_null_deg':float(observed_gap-null.mean()),'p_null_at_least_observed':float((null>=observed_gap).mean())};result['one_random_global_alpha_per_identifier']={'mean_gap_vs_binary_deg':float(random_gap.mean()),'95_interval':[float(np.quantile(random_gap,.025)),float(np.quantile(random_gap,.975))]};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 blocked_gain=gain_np[block_array];blocked_initial=initial.numpy()[block_array];adaptive_held=[];binary_held=[];chosen_alphas=[]
 for held in range(block_array.shape[1]):
  others=[index for index in range(block_array.shape[1]) if index!=held];selection=blocked_gain[:,others].mean(1);adaptive_flat=selection.reshape(len(block_array),-1);adaptive_index=adaptive_flat.argmax(1);adaptive_positive=adaptive_flat[np.arange(len(block_array)),adaptive_index]>0;held_flat=blocked_gain[:,held].reshape(len(block_array),-1);adaptive_actual=np.where(adaptive_positive,held_flat[np.arange(len(block_array)),adaptive_index],0);binary_selection=selection[:,:,-1];binary_component=binary_selection.argmax(1);binary_positive=binary_selection[np.arange(len(block_array)),binary_component]>0;binary_actual=np.where(binary_positive,blocked_gain[:,held,:, -1][np.arange(len(block_array)),binary_component],0);adaptive_held.append(adaptive_actual);binary_held.append(binary_actual);chosen_alphas.extend((adaptive_index%len(adaptive.ALPHAS))[adaptive_positive].tolist())
 adaptive_held=np.stack(adaptive_held,1);binary_held=np.stack(binary_held,1);difference=(adaptive_held-binary_held).mean(1);boot=difference[rng.integers(0,len(difference),(a.replicates,len(difference)))].mean(1);result['cross_detector_replication']={'definition':'select action on two detector seeds and evaluate the same action on the held-out third seed','adaptive_mean_realized_gain_deg':float(adaptive_held.mean()),'binary_mean_realized_gain_deg':float(binary_held.mean()),'adaptive_minus_binary_deg':float(difference.mean()),'bootstrap_95_ci':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],'adaptive_selected_alpha_distribution':{str(float(adaptive.ALPHAS[index])):chosen_alphas.count(index)/len(chosen_alphas) for index in range(len(adaptive.ALPHAS))}};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
