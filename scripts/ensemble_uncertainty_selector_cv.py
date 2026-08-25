#!/usr/bin/env python3
"""Patient-CV selection using detector ensemble disagreement and sensitivity."""
from __future__ import annotations

import argparse, importlib.util, json, random
from pathlib import Path
import torch


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def calibrate(base, learned, risk, data, gain, candidates, score, risk_limit):
    score = score.float()
    value, axis = score.max(1)
    thresholds = torch.cat([torch.tensor([value.min()-1e-3, value.max()+1e-3]),
                            torch.quantile(value, torch.linspace(0, 1, 41))]).unique()
    rows = []
    for threshold in thresholds:
        edit = value > threshold
        prediction = learned.choose(base, data, candidates, axis, edit)
        row = risk.detailed_metrics(base, learned, data, prediction, edit, gain, axis, .02)
        rows.append({"threshold": float(threshold), **row})
    feasible = [row for row in rows if row["joint_harm_rate"] <= risk_limit]
    return max(feasible, key=lambda row: (row["mean_protocol_gain"], row["opportunity_recall"]))


def evaluate(base, selective, learned, risk, cf, rows, outer, args):
    train_rows, cal_rows, test_rows = cf.outer_partitions(rows, outer, args.folds, args.split_seed)
    train, cal, test = [cf.as_data(base, part) for part in [train_rows, cal_rows, test_rows]]
    torch.manual_seed(11000+outer); random.seed(11000+outer)
    proposal = base.train_refiner(train, train, True, args.epochs, image_conditioned=True)
    def state(data):
        gain, proposed, candidates = learned.actual_gains(base, proposal, data)
        sensitivity = selective.sensitivities(base, data["directions"])
        disagreement = data["ensemble_axis_features"][..., 0]
        dispersion = data["ensemble_axis_features"][..., 1]
        scores = {"deviation": disagreement,
                  "dispersion": dispersion,
                  "deviation_x_sensitivity": disagreement * sensitivity,
                  "dispersion_x_sensitivity": dispersion * sensitivity}
        return gain, candidates, scores
    cal_gain, cal_candidates, cal_scores = state(cal)
    test_gain, test_candidates, test_scores = state(test)
    methods = {}
    for name in cal_scores:
        policy = calibrate(base, learned, risk, cal, cal_gain, cal_candidates,
                           cal_scores[name], args.risk_limit)
        value, axis = test_scores[name].float().max(1); edit = value > policy["threshold"]
        prediction = learned.choose(base, test, test_candidates, axis, edit)
        methods[name] = risk.detailed_metrics(base, learned, test, prediction, edit,
                                               test_gain, axis, .02)
        methods[name]["threshold"] = policy["threshold"]
    initial = base.execute(test["directions"]); dummy = test_scores["deviation"].argmax(1)
    methods["no_repair"] = risk.detailed_metrics(base, learned, test, initial,
        torch.zeros(len(initial), dtype=torch.bool), test_gain, dummy, .02)
    oracle_axis, oracle_gain = test_gain.argmax(1), test_gain.max(1).values
    oracle_edit = oracle_gain > 0
    methods["oracle_selector"] = risk.detailed_metrics(base, learned, test,
        learned.choose(base, test, test_candidates, oracle_axis, oracle_edit), oracle_edit,
        test_gain, oracle_axis, .02)
    return {"fold": outer, "methods": methods}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--folds",type=int,default=5); p.add_argument("--split-seed",type=int,default=2027)
    p.add_argument("--epochs",type=int,default=180); p.add_argument("--risk-limit",type=float,default=.10)
    p.add_argument("--results",type=Path,default=Path("/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry/results.json"))
    p.add_argument("--annotations",type=Path,default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/datasets.csv")); p.add_argument("--image-dir",type=Path,default=Path("/media/max/a/caxp/HVAngleEst/HVAngleEst/images"))
    p.add_argument("--output",type=Path,default=Path("outputs/research/ensemble_uncertainty_selector_cv.json")); args=p.parse_args()
    base=load('eu_base','decisive_structured_refinement_benchmark.py'); selective=load('eu_sel','selective_axis_verifier_benchmark.py')
    learned=load('eu_learn','learned_proposal_selective_repair_benchmark.py'); risk=load('eu_risk','risk_adjusted_selector_cv.py'); cf=load('eu_cf','crossfit_risk_selector_cv.py')
    rows=base.load_real_errors(args.results,args.annotations,args.image_dir); folds=[]
    for outer in range(args.folds):
        folds.append(evaluate(base,selective,learned,risk,cf,rows,outer,args))
        args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps({'folds':folds,'aggregate':cf.aggregate(folds)},indent=2)+'\n')
        print(json.dumps({'completed_fold':outer,'methods':folds[-1]['methods']}),flush=True)
    result={'ensemble_component_uncertainty':True,'folds':folds,'aggregate':cf.aggregate(folds)}; args.output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result['aggregate'],indent=2))

if __name__=='__main__': main()
