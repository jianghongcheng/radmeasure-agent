#!/usr/bin/env python3
"""Build generic SQL-edit candidates with exact SQLite counterfactual rewards."""
from __future__ import annotations
import argparse,collections,itertools,json,re,sqlite3,time
from pathlib import Path

ROOT=Path('third_party/spider_data/spider_data');REPO=Path('third_party/spider/evaluation_examples')
def execute(db,sql):
 try:
  connection=sqlite3.connect(f'file:{db}?mode=ro',uri=True);connection.execute('PRAGMA query_only=ON');started=time.perf_counter();connection.set_progress_handler(lambda:int(time.perf_counter()-started>.25),10000);cursor=connection.execute(sql);rows=cursor.fetchmany(10001);connection.close()
  if len(rows)>10000:return {'ok':False,'reason':'too_many_rows'}
  normalized=[tuple(round(x,8) if isinstance(x,float) else x for x in row) for row in rows];return {'ok':True,'columns':len(cursor.description or []),'rows':normalized}
 except Exception as error:return {'ok':False,'reason':type(error).__name__}

def equivalent(left,right):return left.get('ok') and right.get('ok') and left['columns']==right['columns'] and left['rows']==right['rows']

def replace_once(sql,pattern,replacement):
 return re.sub(pattern,replacement,sql,count=1,flags=re.I)

def candidates(sql,columns,question='',limit=40):
 output=[]
 def add(action,candidate):
  candidate=' '.join(candidate.split())
  if candidate.lower()!=sql.strip().lower() and candidate not in {item['sql'] for item in output}:output.append({'action':action,'sql':candidate})
 aggregations=['count','avg','min','max','sum']
 terminal_matches=list(re.finditer(r"'terminal'",sql,re.I))
 if terminal_matches:
  numbers=re.findall(r'(?<!\w)-?\d+(?:\.\d+)?',question);quoted=re.findall(r"['\"]([^'\"]+)['\"]",question);proper=[word for word in re.findall(r'\b[A-Z][A-Za-z_-]*\b',question) if word.lower() not in {'what','which','who','where','when','how','show','list','find','give','return'}];values=[]
  for value in numbers+quoted+proper:
   rendered=value if re.fullmatch(r'-?\d+(?:\.\d+)?',value) else "'"+value.replace("'","''")+"'"
   if rendered not in values:values.append(rendered)
  count=len(terminal_matches)
  assignments=itertools.permutations(values,count) if len(values)>=count else itertools.product(values,repeat=count)
  for assignment in itertools.islice(assignments,24):
   candidate=sql
   for value in assignment:
    if re.search(r"\blike\s*'terminal'",candidate,re.I) and value.startswith("'"):value="'%"+value[1:-1]+"%'"
    candidate=re.sub(r"'terminal'",value,candidate,count=1,flags=re.I)
   add('literal:'+','.join(assignment),candidate)
 for match in list(re.finditer(r'\b(count|avg|min|max|sum)\s*\(',sql,re.I)):
  current=match.group(1).lower()
  for target in aggregations:
   if target!=current:add(f'aggregation:{current}->{target}',sql[:match.start(1)]+target+sql[match.end(1):])
 if re.search(r'\bdesc\b',sql,re.I):add('order:desc->asc',replace_once(sql,r'\bdesc\b','asc'))
 if re.search(r'\basc\b',sql,re.I):add('order:asc->desc',replace_once(sql,r'\basc\b','desc'))
 if re.search(r'\border\s+by\b',sql,re.I) and not re.search(r'\b(asc|desc)\b',sql,re.I):
  suffix=re.search(r'\blimit\b',sql,re.I);position=suffix.start() if suffix else len(sql);add('order:add_desc',sql[:position]+' DESC '+sql[position:]);add('order:add_asc',sql[:position]+' ASC '+sql[position:])
 if re.search(r'^\s*select\s+distinct\b',sql,re.I):add('distinct:remove',re.sub(r'^(\s*select\s+)distinct\s+',r'\1',sql,count=1,flags=re.I))
 else:add('distinct:add',re.sub(r'^(\s*select\s+)',r'\1DISTINCT ',sql,count=1,flags=re.I))
 limit_match=re.search(r'\blimit\s+(\d+)',sql,re.I)
 if limit_match:
  add('limit:remove',sql[:limit_match.start()]+sql[limit_match.end():])
  for value in [1,3,5,10]:
   if value!=int(limit_match.group(1)):add(f'limit:{value}',sql[:limit_match.start(1)]+str(value)+sql[limit_match.end(1):])
 for match in list(re.finditer(r'(?<![<>!])=(?!=)|>=|<=|!=|(?<![<>=])>(?!=)|(?<![<>=])<(?!=)',sql)):
  current=match.group(0)
  for target in ['=','!=','>','<','>=','<=']:
   if target!=current:add(f'operator:{current}->{target}',sql[:match.start()]+target+sql[match.end():])
 for match in list(re.finditer(r'\b(and|or)\b',sql,re.I)):
  current=match.group(1).lower();target='or' if current=='and' else 'and';add(f'logic:{current}->{target}',sql[:match.start(1)]+target+sql[match.end(1):])
 lowered=sql.lower();used=[column for column in columns if column!='*' and re.search(rf'\b{re.escape(column.lower())}\b',lowered)]
 alternatives=[column for column in columns if column!='*']
 for current in used[:4]:
  for target in alternatives[:12]:
   if target.lower()!=current.lower():add(f'column:{current}->{target}',re.sub(rf'\b{re.escape(current)}\b',target,sql,count=1,flags=re.I))
 return output[:limit]

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--repo',type=Path,default=REPO);p.add_argument('--data',type=Path);p.add_argument('--predictions',type=Path);p.add_argument('--source');p.add_argument('--limit',type=int,default=40);p.add_argument('--output',type=Path,default=Path('outputs/research/spider_executable_edits.json'));a=p.parse_args();data_path=a.data or a.root/'dev.json';prediction_path=a.predictions or a.repo/'pred_example.txt';source=a.source or f'{data_path} plus {prediction_path}';dev=json.load(open(data_path));tables=json.load(open(a.root/'tables.json'));schema={row['db_id']:[name for _,name in row['column_names_original']] for row in tables};predictions=prediction_path.read_text().splitlines();assert len(dev)==len(predictions);records=[];counts=collections.Counter()
 for index,(example,predicted) in enumerate(zip(dev,predictions)):
  db_id=example['db_id'];db=a.root/'database'/db_id/f'{db_id}.sqlite';gold_result=execute(db,example['query']);base_result=execute(db,predicted);base_correct=equivalent(base_result,gold_result);edits=[]
  for item in candidates(predicted,schema[db_id],example['question'],a.limit):
   result=execute(db,item['sql']);correct=equivalent(result,gold_result);edits.append({**item,'executable':bool(result.get('ok')),'correct':correct,'advantage':int(correct)-int(base_correct)})
  records.append({'index':index,'db_id':db_id,'question':example['question'],'gold_sql':example['query'],'predicted_sql':predicted,'base_executable':bool(base_result.get('ok')),'base_correct':base_correct,'candidates':edits});counts.update({'cases':1,'base_correct':base_correct,'base_executable':bool(base_result.get('ok')),'candidates':len(edits),'beneficial_candidates':sum(e['advantage']>0 for e in edits),'harmful_candidates':sum(e['advantage']<0 for e in edits),'oracle_recoverable':any(e['advantage']>0 for e in edits)})
  if (index+1)%100==0:
   partial={'source':source,'exact_counterfactual_reward':'SQLite execution-result equality','partial':True,'records':records};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(partial,indent=2)+'\n');print(json.dumps({'completed':index+1}),flush=True)
 summary={key:int(value) for key,value in counts.items()};summary['baseline_execution_accuracy']=counts['base_correct']/counts['cases'];summary['oracle_recoverable_rate']=counts['oracle_recoverable']/counts['cases'];result={'source':source,'prediction_sha256':__import__('hashlib').sha256(prediction_path.read_bytes()).hexdigest(),'exact_counterfactual_reward':'SQLite execution-result equality','summary':summary,'records':records};a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
