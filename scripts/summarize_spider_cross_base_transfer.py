#!/usr/bin/env python3
"""Create a compact, auditable table from existing Spider cross-base runs."""
from __future__ import annotations
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/"outputs"/"research"

def main():
    cross=json.loads((OUT/"spider_cross_base_selector_cv.json").read_text())
    rows=[]
    for base in ("1b","3b","7b","15b"):
        within=json.loads((OUT/f"spider_codes_{base}_selector_cv.json").read_text())["aggregate"]
        target=cross["targets"][base]["pooled"]
        rows.append({"base":f"CodeS-{base.upper()}","base_execution_accuracy":target["no_repair"]["execution_accuracy"],
                     "oracle_headroom":target["oracle_candidate"]["absolute_gain"],
                     "within_base_learned_gain":within["learned_exact_advantage"]["net_benefit_minus_harm"]["mean"],
                     "leave_one_base_out_gain":target["learned_cross_base"]["absolute_gain"],
                     "cross_base_benefit_count":target["learned_cross_base"]["benefit_count"],
                     "cross_base_harm_count":target["learned_cross_base"]["harm_count"]})
    with (OUT/"spider_cross_base_transfer_summary.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/"spider_cross_base_transfer_summary.json").write_text(json.dumps({"source":"existing exploratory database-grouped runs; no new model training", "rows":rows},indent=2)+"\n")
    print(json.dumps(rows,indent=2))
if __name__=="__main__":main()
