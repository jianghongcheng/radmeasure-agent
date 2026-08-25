#!/usr/bin/env python3
"""Audit and merge the five clean-base OOF prediction folds."""
from __future__ import annotations
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'third_party/spider_data/spider_data';OUT=ROOT/'outputs/research/spider_clean_base'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 seal=json.loads((ROOT/'outputs/research/spider_protocol_seal_v1.json').read_text());development=set(seal['development_databases']);confirmatory=set(seal['confirmatory_databases']);all_rows=json.loads((DATA/'train_spider.json').read_text())+json.loads((DATA/'train_others.json').read_text());expected={index:row for index,row in enumerate(all_rows) if row['db_id'] in development};predictions={};fold_hashes={};test_dbs=[]
 for fold in range(5):
  path=OUT/f'fold{fold}_predictions.json';artifact=json.loads(path.read_text());fold_hashes[str(fold)]=sha(path);assert artifact['confirmatory_gold_used'] is False;assert not(set(artifact['test_databases'])&confirmatory);test_dbs.extend(artifact['test_databases'])
  for row in artifact['predictions']:
   index=row['source_index'];assert index not in predictions;assert expected[index]['db_id']==row['db_id'];assert expected[index]['question']==row['question'];predictions[index]=row['predicted_sql']
 assert set(predictions)==set(expected);assert len(test_dbs)==len(set(test_dbs))==len(development)
 ordered=sorted(expected);data_path=OUT/'oof_development_data.json';prediction_path=OUT/'oof_predictions.txt';data_path.write_text(json.dumps([expected[index] for index in ordered],indent=2)+'\n');prediction_path.write_text('\n'.join(predictions[index].replace('\n',' ') for index in ordered)+'\n');result={'n':len(ordered),'development_databases':len(development),'confirmatory_overlap':0,'every_example_predicted_once':True,'every_database_tested_once':True,'fold_prediction_sha256':fold_hashes,'data_sha256':sha(data_path),'predictions_sha256':sha(prediction_path),'confirmatory_gold_used':False};(OUT/'oof_merge_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
