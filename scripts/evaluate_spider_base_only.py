#!/usr/bin/env python3
"""Evaluate only base executability/accuracy; deliberately generates no candidates."""
from __future__ import annotations
import argparse,collections,importlib.util,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'third_party/spider_data/spider_data'
def load_builder():
 path=ROOT/'scripts/build_spider_executable_edits.py';spec=importlib.util.spec_from_file_location('builder',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def main():
 p=argparse.ArgumentParser();p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();artifact=json.loads(a.predictions.read_text());predictions=artifact['predictions'];all_rows=json.loads((DATA/'train_spider.json').read_text())+json.loads((DATA/'train_others.json').read_text());builder=load_builder();counts=collections.Counter();rows=[]
 for prediction in predictions:
  example=all_rows[prediction['source_index']];assert example['db_id']==prediction['db_id'];db=DATA/'database'/example['db_id']/f"{example['db_id']}.sqlite";gold=builder.execute(db,example['query']);pred=builder.execute(db,prediction['predicted_sql']);correct=builder.equivalent(pred,gold);counts.update({'cases':1,'executable':bool(pred.get('ok')),'correct':correct});rows.append({'source_index':prediction['source_index'],'db_id':example['db_id'],'executable':bool(pred.get('ok')),'correct':correct})
 result={'base_only_evaluation':True,'candidate_generation_performed':False,'confirmatory_gold_used':False,'source_prediction_file':str(a.predictions),'cases':counts['cases'],'execution_accuracy':counts['correct']/counts['cases'],'executable_rate':counts['executable']/counts['cases'],'rows':rows};a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({key:value for key,value in result.items() if key!='rows'},indent=2))
if __name__=='__main__':main()
