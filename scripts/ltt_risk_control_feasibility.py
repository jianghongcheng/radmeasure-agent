#!/usr/bin/env python3
"""Power audit for finite-family and nested-path Bernoulli harm control.

The nested-path calculation is an optimistic sensitivity analysis.  It applies
only when the candidate policies are a single, pointwise nested threshold path
whose risk test can exploit that ordering.  It must not be used for an
arbitrary collection of policy architectures, scores, or action rules.
"""
import argparse,importlib.util,json,math
from pathlib import Path

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def main():
 p=argparse.ArgumentParser();p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);p.add_argument('--policies',type=int,default=20);p.add_argument('--delta',type=float,default=.05);p.add_argument('--results',type=Path,default=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json'));p.add_argument('--annotations',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv'));p.add_argument('--image-dir',type=Path,default=Path('/media/max/a/caxp/HVAngleEst/HVAngleEst/images'));p.add_argument('--output',type=Path,default=Path('outputs/research/ltt_risk_control_feasibility.json'));a=p.parse_args();base=load('lf_base','decisive_structured_refinement_benchmark.py');cf=load('lf_cf','crossfit_risk_selector_cv.py');rows=base.load_real_errors(a.results,a.annotations,a.image_dir);sizes=[]
 for outer in range(a.folds):
  _,calibration,_=cf.outer_partitions(rows,outer,a.folds,a.seed);sizes.append(len({row['patient_id'] for row in calibration}))
 adjusted=a.delta/a.policies
 epsilons=[.01,.02,.05,.1,.15]
 minimum_n={str(epsilon):math.ceil(math.log(adjusted)/math.log(1-epsilon)) for epsilon in epsilons}
 nested_minimum_n={str(epsilon):math.ceil(math.log(a.delta)/math.log(1-epsilon)) for epsilon in epsilons}
 folds=[]
 for fold,n in enumerate(sizes):
  folds.append({
   'fold':fold,
   'n_calibration_identifiers':n,
   'finite_family_smallest_certifiable_risk_with_zero_harms':1-adjusted**(1/n),
   'single_nested_path_smallest_certifiable_risk_with_zero_harms':1-a.delta**(1/n),
  })
 result={
  'familywise_delta':a.delta,
  'number_of_predeclared_policies':a.policies,
  'bonferroni_level_per_policy':adjusted,
  'calibration_folds':folds,
  'finite_family_minimum_identifiers_with_zero_observed_harms':minimum_n,
  'single_nested_path_sensitivity':{
   'status':'optimistic sensitivity only; nesting is not established by the current finite-family audit',
   'minimum_identifiers_with_zero_observed_harms':nested_minimum_n,
  },
  'interpretation':(
   'For M arbitrary policies, zero harms certifies risk epsilon when '
   '(1-epsilon)^n <= delta/M. If all policies instead form one valid nested '
   'threshold path, the optimistic no-multiplicity sensitivity uses '
   '(1-epsilon)^n <= delta. The current script audits sample-size feasibility '
   'and does not establish policy nesting.'
  ),
 }
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
