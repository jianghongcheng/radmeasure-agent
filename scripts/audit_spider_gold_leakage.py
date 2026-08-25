#!/usr/bin/env python3
"""Reproducible audit that Spider candidates/features cannot access gold labels."""
from __future__ import annotations
import argparse,copy,hashlib,importlib.util,json
from pathlib import Path

def load(name,file):
 spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--records',type=Path,default=Path('outputs/research/spider_executable_edits.json'));p.add_argument('--tables',type=Path,default=Path('third_party/spider_data/spider_data/tables.json'));p.add_argument('--output',type=Path,default=Path('outputs/research/spider_gold_leakage_audit.json'));p.add_argument('--folds',type=int,default=5);p.add_argument('--seed',type=int,default=2027);a=p.parse_args();builder=load('audit_builder','build_spider_executable_edits.py');selector=load('audit_selector','spider_advantage_selector_cv.py');records=json.load(open(a.records))['records'];tables=json.load(open(a.tables));schema={row['db_id']:[name for _,name in row['column_names_original']] for row in tables};mismatches=[];feature_mismatches=[];label_errors=[];candidate_count=0;original_corpus=[];regenerated_corpus=[]
 for record in records:
  original=[{'action':candidate['action'],'sql':candidate['sql']} for candidate in record['candidates']];regenerated=builder.candidates(record['predicted_sql'],schema[record['db_id']],record['question'],40);candidate_count+=len(original);original_corpus.extend([(record['index'],row) for row in original]);regenerated_corpus.extend([(record['index'],row) for row in regenerated])
  if original!=regenerated:mismatches.append(record['index'])
  for candidate in record['candidates']:
   expected=int(candidate['correct'])-int(record['base_correct'])
   if candidate['advantage']!=expected:label_errors.append((record['index'],candidate['action']))
   stripped_record={key:value for key,value in record.items() if key not in {'gold_sql','base_correct'}};stripped_candidate={key:value for key,value in candidate.items() if key not in {'correct','advantage'}};before=selector.text(record,candidate);after=selector.text(stripped_record,stripped_candidate)
   if before!=after:feature_mismatches.append((record['index'],candidate['action']))
 folds=[];dbs={record['db_id'] for record in records}
 for outer in range(a.folds):
  test={db for db in dbs if selector.fold_of(db,a.folds,a.seed)==outer};calibration={db for db in dbs if selector.fold_of(db,a.folds,a.seed)==(outer+1)%a.folds};training=dbs-test-calibration;folds.append({'fold':outer,'train_databases':len(training),'calibration_databases':len(calibration),'test_databases':len(test),'pairwise_disjoint':not(training&calibration or training&test or calibration&test)})
 mutated=copy.deepcopy(records[0]);mutated['gold_sql']='SELECT deliberately_mutated_gold';mutated['base_correct']=not mutated['base_correct']
 for candidate in mutated['candidates']:candidate['correct']=not candidate['correct'];candidate['advantage']=-candidate['advantage']
 mutation_feature_invariant=all(selector.text(records[0],left)==selector.text(mutated,right) for left,right in zip(records[0]['candidates'],mutated['candidates']))
 result={'audit_passed':not mismatches and not feature_mismatches and not label_errors and all(row['pairwise_disjoint'] for row in folds) and mutation_feature_invariant,'candidate_generator_inputs':['predicted_sql','question','public_schema_column_names','candidate_limit'],'candidate_generator_excludes':['gold_sql','gold_execution_result','candidate_correctness','advantage_label'],'candidate_count':candidate_count,'regenerated_exact_match':not mismatches,'candidate_corpus_sha256':digest(original_corpus),'regenerated_corpus_sha256':digest(regenerated_corpus),'inference_feature_gold_label_invariant':not feature_mismatches and mutation_feature_invariant,'advantage_label_identity_valid':not label_errors,'database_split_audit':folds,'mismatch_examples':mismatches[:10],'feature_mismatch_examples':feature_mismatches[:10],'label_error_examples':label_errors[:10]};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
