#!/usr/bin/env python3
"""Freeze synthetic predictions at preregistered coordinates without reading outcomes."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PREREG=ROOT/'protocols/SYNTHETIC_PHASE_EXTENSION_PREREGISTRATION_V1.json'
SYNTH=ROOT/'outputs/research/synthetic_selective_phase_extended_v1.json'
OUTPUT=ROOT/'outputs/research/synthetic_phase_extended_v1_predictions_only.json'

def interpolate(summary,clusters,precision,noise,metric):
 cs=sorted({int(r['cluster_count']) for r in summary});ps=sorted({float(r['proposal_precision']) for r in summary});ns=sorted({float(r['label_noise']) for r in summary})
 assert cs[0]<=clusters<=cs[-1] and ps[0]<=precision<=ps[-1] and ns[0]<=noise<=ns[-1]
 def bracket(values,value,log=False):
  if value in values:return value,value,0.
  hi=next(v for v in values if v>value);lo=values[values.index(hi)-1];f=np.log if log else lambda x:x
  return lo,hi,float((f(value)-f(lo))/(f(hi)-f(lo)))
 c0,c1,wc=bracket(cs,clusters,True);p0,p1,wp=bracket(ps,precision);n0,n1,wn=bracket(ns,noise)
 lookup={(int(r['cluster_count']),float(r['proposal_precision']),float(r['label_noise'])):float(r[metric]) for r in summary};value=0.
 for c,a in [(c0,1-wc),(c1,wc)] if c0!=c1 else [(c0,1.)]:
  for p,b in [(p0,1-wp),(p1,wp)] if p0!=p1 else [(p0,1.)]:
   for n,d in [(n0,1-wn),(n1,wn)] if n0!=n1 else [(n0,1.)]:value+=a*b*d*lookup[(c,p,n)]
 return value

def predict(summary,row,noise):
 args=(row['clusters'],row['proposal_precision'],noise)
 gain=interpolate(summary,*args,'learned_gain_mean')
 best=interpolate(summary,*args,'learned_minus_best_constant_mean')
 p_noop=interpolate(summary,*args,'prob_learned_beats_no_op')
 p_all=interpolate(summary,*args,'prob_learned_beats_apply_all')
 apply_all=2*row['proposal_precision']-1
 margin_all=gain-apply_all
 valuable=gain>0 and margin_all>0 and p_noop>=.8 and p_all>=.8
 return {'noise':noise,'predicted_selective_gain':gain,'predicted_margin_vs_no_op':gain,'predicted_margin_vs_apply_all':margin_all,'predicted_margin_vs_best_constant':best,'probability_beats_no_op':p_noop,'probability_beats_apply_all':p_all,'categorical_selective_value':valuable}

def main():
 prereg=json.loads(PREREG.read_text());summary=json.loads(SYNTH.read_text())['summary'];predictions=[]
 for row in prereg['frozen_real_coordinates_without_outcomes']:
  primary=predict(summary,row,row['advantage_noise_primary'])
  sensitivity=[predict(summary,row,n) for n in row.get('noise_sensitivity',[])]
  predictions.append({'domain':row['domain'],'clusters':row['clusters'],'proposal_precision':row['proposal_precision'],'primary':primary,'sensitivity':sensitivity})
 payload={'status':'predictions_frozen_before_outcome_join','preregistration_sha256':'b2a40c418b4e04e1bb2118134eec6591f624977f41324d66a625be6044b01065','synthetic_artifact_sha256':hashlib.sha256(SYNTH.read_bytes()).hexdigest(),'predictions':predictions}
 OUTPUT.write_text(json.dumps(payload,indent=2)+'\n');digest=hashlib.sha256(OUTPUT.read_bytes()).hexdigest();OUTPUT.with_suffix('.sha256').write_text(f'{digest}  {OUTPUT.name}\n');print(json.dumps({'prediction_file':str(OUTPUT),'sha256':digest,'predictions':predictions},indent=2))
if __name__=='__main__':main()
