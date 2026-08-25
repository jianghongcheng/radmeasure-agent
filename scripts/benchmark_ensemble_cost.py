#!/usr/bin/env python3
"""Measure sequential detector-forward cost for the ensemble-size ablation."""
from __future__ import annotations
import argparse, importlib.util, json, statistics, sys, time
from pathlib import Path
import torch

SOURCE=Path('/media/max/a/caxp (Copy 2)/scripts/benchmarks')
CHECKPOINTS=Path('/media/max/a/caxp (Copy 2)/outputs/benchmarks/controlled_cross_model_v2/hvangle_axis_geometry')

def load_source():
 sys.path.insert(0,str(SOURCE));spec=importlib.util.spec_from_file_location('hvaxis',SOURCE/'hvangle_axis_geometry_benchmark.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def synchronize(device):
 if device.type=='cuda':torch.cuda.synchronize()

def main():
 p=argparse.ArgumentParser();p.add_argument('--repeats',type=int,default=20);p.add_argument('--warmup',type=int,default=3);p.add_argument('--output',type=Path,default=Path('outputs/research/ensemble_inference_cost.json'));a=p.parse_args();module=load_source();device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');seeds=[17,42,73];models=[]
 for seed in seeds:
  model=module.AxisUNet(32).to(device);model.load_state_dict(torch.load(CHECKPOINTS/f'hvangle_axis_seed{seed}.pt',map_location=device));model.eval();models.append(model)
 sample=torch.zeros(1,3,256,256,device=device)
 rows=[]
 with torch.inference_mode():
  for size in (1,2,3):
   for _ in range(a.warmup):
    for model in models[:size]:model(sample)
   synchronize(device);times=[]
   for _ in range(a.repeats):
    started=time.perf_counter()
    for model in models[:size]:model(sample)
    synchronize(device);times.append((time.perf_counter()-started)*1000)
   rows.append({'ensemble_size':size,'device':str(device),'batch_size':1,'image_size':256,'parameters_total':sum(sum(p.numel() for p in m.parameters()) for m in models[:size]),'checkpoint_megabytes_total':sum((CHECKPOINTS/f'hvangle_axis_seed{s}.pt').stat().st_size for s in seeds[:size])/1e6,'latency_ms_mean':statistics.mean(times),'latency_ms_sd':statistics.stdev(times),'throughput_images_per_second':1000/statistics.mean(times)})
 result={'forward_only':True,'sequential_execution':True,'torch_threads':torch.get_num_threads(),'rows':rows};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
