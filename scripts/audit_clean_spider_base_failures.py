#!/usr/bin/env python3
"""Classify clean-base failures before interpreting selector experiments."""
from __future__ import annotations
import argparse,collections,json,random,re,sqlite3,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];DBROOT=ROOT/'third_party/spider_data/spider_data/database'
def diagnose(db_id,sql):
 text=sql.strip();lower=text.lower()
 if not text:return 'parse_failure:empty','empty output'
 if '```' in text:return 'parse_failure:markdown_fence','markdown fence'
 if not re.match(r'^(select|with)\b',lower):return 'parse_failure:non_sql_prefix','output does not begin with SELECT/WITH'
 if ';' in text.rstrip(';'):return 'parse_failure:multiple_statements','multiple statements or trailing explanation'
 db=DBROOT/db_id/f'{db_id}.sqlite'
 try:
  connection=sqlite3.connect(f'file:{db}?mode=ro',uri=True);connection.execute('PRAGMA query_only=ON');started=time.perf_counter();connection.set_progress_handler(lambda:int(time.perf_counter()-started>.5),10000);connection.execute(text).fetchmany(10001);connection.close();return 'executable_but_wrong','query executes'
 except Exception as error:
  message=str(error)
  if 'no such table' in message:return 'execution_failure:no_such_table',message
  if 'no such column' in message:return 'execution_failure:no_such_column',message
  if 'ambiguous column' in message:return 'execution_failure:ambiguous_column',message
  if 'syntax error' in message or 'incomplete input' in message or 'unrecognized token' in message:return 'execution_failure:syntax',message
  if 'no such function' in message or 'wrong number of arguments' in message:return 'execution_failure:function',message
  if 'interrupted' in message:return 'execution_failure:timeout',message
  return 'execution_failure:other',f'{type(error).__name__}: {message}'
def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=ROOT/'outputs/research/spider_clean_base_executable_edits.json');p.add_argument('--sample',type=int,default=50);p.add_argument('--seed',type=int,default=20260824);p.add_argument('--output',type=Path,default=ROOT/'outputs/research/spider_clean_base_failure_audit.json');a=p.parse_args();artifact=json.loads(a.input.read_text())
 if 'records' in artifact:
  records=artifact['records']
 else:
  data=json.loads((ROOT/'third_party/spider_data/spider_data/train_spider.json').read_text())+json.loads((ROOT/'third_party/spider_data/spider_data/train_others.json').read_text())
  records=[]
  for prediction in artifact['predictions']:
   example=data[prediction['source_index']]
   records.append({'index':prediction['source_index'],'db_id':example['db_id'],'question':example['question'],'gold_sql':example['query'],'predicted_sql':prediction['predicted_sql'],'base_correct':False})
 failures=[]
 for row in records:
  category,message=diagnose(row['db_id'],row['predicted_sql'])
  if row.get('base_correct') or category=='executable_but_wrong' and 'base_correct' not in row:
   # Prediction-only artifacts need execution equivalence to distinguish correct rows;
   # the base-only evaluator already records this by source index.
   pass
  failures.append((row,category,message))
 if 'records' not in artifact:
  base_path=a.input.with_name(a.input.stem.replace('_predictions','_base_only')+'.json')
  correctness={row['source_index']:row['correct'] for row in json.loads(base_path.read_text())['rows']}
  failures=[item for item in failures if not correctness[item[0]['index']]]
 else:
  failures=[item for item in failures if not item[0]['base_correct']]
 rows=[];counts=collections.Counter()
 for row,category,message in failures:
  counts[category]+=1;rows.append({'index':row['index'],'db_id':row['db_id'],'question':row['question'],'gold_sql':row['gold_sql'],'predicted_sql':row['predicted_sql'],'category':category,'sqlite_message':message})
 rng=random.Random(a.seed);sample=sorted(rng.sample(rows,min(a.sample,len(rows))),key=lambda row:row['index']);sample_counts=collections.Counter(row['category'] for row in sample);result={'population':{'cases':len(records),'base_failures':len(failures),'category_counts':dict(counts),'category_rates_among_failures':{key:value/len(failures) for key,value in counts.items()}},'sample':{'seed':a.seed,'n':len(sample),'category_counts':dict(sample_counts),'cases':sample}};a.output.write_text(json.dumps(result,indent=2)+'\n');md=['# Clean Spider base: fixed 50-failure audit','',f'Seed: `{a.seed}`',f'Population failures: {len(failures)} / {len(records)}','', '## Sample classification','']
 for category,count in sorted(sample_counts.items()):md.append(f'- {category}: {count}')
 for number,row in enumerate(sample,1):md.extend(['',f'## {number}. index {row["index"]} — {row["category"]}','',f'- DB: `{row["db_id"]}`',f'- SQLite: `{row["sqlite_message"]}`',f'- Question: {row["question"]}',f'- Prediction: `{row["predicted_sql"]}`',f'- Gold: `{row["gold_sql"]}`'])
 a.output.with_suffix('.md').write_text('\n'.join(md)+'\n');print(json.dumps({'population':result['population'],'sample_counts':dict(sample_counts)},indent=2))
if __name__=='__main__':main()
