#!/usr/bin/env python3
"""Plot magnitude-aware MAE/harm frontiers and export nondominated points."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def nondominated(points,metric):
 ordered=sorted(points,key=lambda p:(p['row_level'][metric],p['row_level']['mean_MAE']));result=[];best=float('inf')
 for point in ordered:
  mae=point['row_level']['mean_MAE']
  if mae<best-1e-9:result.append(point);best=mae
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=Path('outputs/research/pareto_frontier_cv.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/pareto_frontier.png'));p.add_argument('--points-output',type=Path,default=Path('outputs/research/pareto_nondominated_points.json'));a=p.parse_args();data=json.load(open(a.input))['points']
 definitions={
  'Continuous geometry':lambda x:x['family']=='all_geometry',
  'Continuous output':lambda x:x['family']=='all_output',
  'Learned binary gate':lambda x:x['family']=='learned_geometry' and x.get('alpha')==1,
  'Learned gate + geometry shrink':lambda x:x['family']=='learned_geometry',
  'Learned gate + output shrink':lambda x:x['family']=='learned_output'}
 colors=['#777777','#222222','#d95f02','#1b9e77','#7570b3'];fig,axes=plt.subplots(1,2,figsize=(11,4.2));export={}
 for ax,tau in zip(axes,[.5,1.]):
  metric=f'harm_at_{tau:g}deg';export[metric]={}
  for (label,predicate),color in zip(definitions.items(),colors):
   frontier=nondominated([x for x in data if predicate(x)],metric);export[metric][label]=frontier;xs=[100*x['row_level'][metric] for x in frontier];ys=[x['row_level']['mean_MAE'] for x in frontier];ax.plot(xs,ys,marker='o',ms=3,label=label,color=color)
  no_repair=next(x for x in data if x['family']=='no_repair');ax.scatter([0],[no_repair['row_level']['mean_MAE']],marker='*',s=100,color='black',zorder=5,label='No repair');ax.set_xlabel(f'Harm@{tau:g}° (%)');ax.set_ylabel('Mean absolute error (°)');ax.set_title(f'Magnitude-aware harm threshold τ={tau:g}°');ax.grid(alpha=.25)
 axes[1].legend(fontsize=8,loc='best');fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=220);fig.savefig(a.output.with_suffix('.pdf'));a.points_output.write_text(json.dumps(export,indent=2)+'\n');print(a.output)
if __name__=='__main__':main()
