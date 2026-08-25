#!/usr/bin/env python3
"""Join already-hashed predictions to historical outcomes and render extended figures."""
from __future__ import annotations
import importlib.util,json,math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/research'
def load_module():
 p=ROOT/'scripts/map_real_domains_to_synthetic_phase.py';s=importlib.util.spec_from_file_location('phase_map',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def main():
 prediction_path=OUT/'synthetic_phase_extended_v1_predictions_only.json'
 predictions=json.loads(prediction_path.read_text())['predictions'];module=load_module()
 domains=[module.xray_coordinate(),*module.spider_coordinates()]
 observed={
  'X-ray':{'gain':domains[0]['observed_gain'],'categorical_selective_value':False,'criterion':'negative MAE reduction'},
  'Spider weak':{'gain':domains[1]['observed_gain'],'categorical_selective_value':True,'criterion':'positive 11.51-point gain and apply-all is worse than no-op'},
  'Spider clean':{'gain':domains[2]['observed_gain'],'categorical_selective_value':False,'criterion':'database-clustered CI crosses zero'},
 }
 rows=[]
 for prediction in predictions:
  truth=observed[prediction['domain']];predicted=prediction['primary']['categorical_selective_value']
  rows.append({'domain':prediction['domain'],'prediction':predicted,'observed':truth['categorical_selective_value'],'direction_match':predicted==truth['categorical_selective_value'],'observed_gain':truth['gain'],'observed_criterion':truth['criterion'],'primary':prediction['primary'],'sensitivity':prediction['sensitivity']})
 disagreement=domains[0]['advantage_noise'];latent_q=(1-math.sqrt(max(0,1-2*disagreement)))/2
 payload={'prediction_sha256':'8e8f3b43c9cfb4f99c66cb1443ad8c0e04927a7f945ac6e012e4711f89bafe84','rows':rows,'xray_noise_mapping':{'observed_pairwise_disagreement_D':disagreement,'independent_symmetric_flip_relation':'D = 2 q (1-q)','implied_latent_flip_q':latent_q,'warning':'Direct disagreement and simulator label-flip probability are not scale-identical. At registered noise 0.4 the synthetic categorical conclusion reverses, so X-ray agreement is not robust.'}}
 (OUT/'synthetic_phase_extended_v1_revealed_comparison.json').write_text(json.dumps(payload,indent=2)+'\n')
 synthetic=json.loads((OUT/'synthetic_selective_phase_extended_v1.json').read_text());summary=synthetic['summary'];module.enrich(domains,summary)
 module.figure1(OUT/'figure1_extended_real_domains_on_synthetic_phase.png',domains,summary)
 module.figure2(OUT/'figure2_extended_selective_strategy_boundaries.png',summary)
 print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
