#!/usr/bin/env python3
"""Zero-learning Spider baseline: edit only invalid bases using execution alone.

The policy never inspects gold SQL or candidate correctness. If the base query
does not execute, it takes the first candidate in the frozen operator order that
does execute; otherwise it returns the base unchanged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"/"research"

FILES={
    "Spider weak":OUT/"spider_executable_edits.json",
    "Spider clean":OUT/"spider_clean_base_executable_edits.json",
    "CodeS-1B":OUT/"spider_codes_1b_executable_edits.json",
    "CodeS-3B":OUT/"spider_codes_3b_executable_edits.json",
    "CodeS-7B":OUT/"spider_codes_7b_executable_edits.json",
    "CodeS-15B":OUT/"spider_codes_15b_executable_edits.json",
}

def clustered_ci(cases, seed=2027, repeats=10000):
    grouped={}
    for row in cases: grouped.setdefault(row["db_id"],[]).append(row)
    dbs=sorted(grouped);rng=np.random.default_rng(seed)
    sizes=np.array([len(grouped[db]) for db in dbs])
    net=np.array([sum(int(not x["before_correct"] and x["after_correct"])-int(x["before_correct"] and not x["after_correct"]) for x in grouped[db]) for db in dbs])
    sampled=rng.integers(0,len(dbs),size=(repeats,len(dbs)))
    values=net[sampled].sum(1)/sizes[sampled].sum(1)
    return [float(np.quantile(values,.025)),float(np.quantile(values,.975))]

def evaluate(records):
    cases=[]
    for row in records:
        chosen=None
        if not row["base_executable"]:
            chosen=next((candidate for candidate in row["candidates"] if candidate["executable"]),None)
        after=bool(chosen["correct"]) if chosen is not None else bool(row["base_correct"])
        cases.append({"index":row["index"],"db_id":row["db_id"],"base_executable":bool(row["base_executable"]),
                      "before_correct":bool(row["base_correct"]),"edited":chosen is not None,
                      "after_correct":after,"chosen_action":chosen["action"] if chosen else None})
    n=len(cases);benefit=sum(not x["before_correct"] and x["after_correct"] for x in cases);harm=sum(x["before_correct"] and not x["after_correct"] for x in cases)
    invalid=sum(not x["base_executable"] for x in cases);covered=sum(x["edited"] for x in cases)
    return {"n":n,"base_accuracy":sum(x["before_correct"] for x in cases)/n,
            "execution_only_accuracy":sum(x["after_correct"] for x in cases)/n,
            "absolute_gain":(benefit-harm)/n,"benefit_count":benefit,"harm_count":harm,
            "invalid_base_count":invalid,"invalid_base_rate":invalid/n,
            "invalid_with_executable_candidate_count":covered,"coverage":covered/n,
            "repair_precision_among_triggered":benefit/covered if covered else 0.0,
            "database_clustered_95pct_ci":clustered_ci(cases),"cases":cases}

def combine_with_selector(execution_result, selector_payload):
    selector_cases={row["index"]:row for fold in selector_payload["folds"]
                    for row in fold["methods"]["learned_selector_learned_proposal"]["cases"]}
    cases=[]
    for execution in execution_result["cases"]:
        selector=selector_cases[execution["index"]]
        use_execution=execution["edited"]
        cases.append({"index":execution["index"],"db_id":execution["db_id"],
                      "before_correct":execution["before_correct"],
                      "after_correct":execution["after_correct"] if use_execution else selector["after_correct"],
                      "edited":bool(use_execution or selector["edited"]),
                      "source":"execution_only" if use_execution else ("learned_selector" if selector["edited"] else "stop")})
    n=len(cases); benefit=sum(not x["before_correct"] and x["after_correct"] for x in cases);harm=sum(x["before_correct"] and not x["after_correct"] for x in cases)
    return {"n":n,"execution_accuracy":sum(x["after_correct"] for x in cases)/n,
            "absolute_gain":(benefit-harm)/n,"benefit_count":benefit,"harm_count":harm,
            "execution_stage_count":sum(x["source"]=="execution_only" for x in cases),
            "selector_stage_count":sum(x["source"]=="learned_selector" for x in cases),
            "database_clustered_95pct_ci":clustered_ci(cases),"cases":cases}

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT/"spider_execution_only_baseline.json");args=parser.parse_args()
    results={name:evaluate(json.loads(path.read_text())["records"]) for name,path in FILES.items()}
    selector=json.loads((OUT/"spider_clean_base_quadruple_cv.json").read_text())
    cascade=combine_with_selector(results["Spider clean"],selector)
    payload={"policy":"If and only if base execution fails, use first executable frozen candidate; no learned scores or gold at decision time.",
             "cascade_policy":"Use execution-only repair when it triggers; otherwise retain the existing cross-fitted learned-selector decision.",
             "results":results,"spider_clean_execution_then_selector":cascade}
    args.output.write_text(json.dumps(payload,indent=2)+"\n")
    printable={name:{k:v for k,v in result.items() if k!="cases"} for name,result in results.items()}
    printable["Spider clean: execution then selector"]={k:v for k,v in cascade.items() if k!="cases"}
    print(json.dumps(printable,indent=2))

if __name__=="__main__":main()
