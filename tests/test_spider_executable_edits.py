import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def module(name):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py');value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def test_terminal_literal_candidates_are_grounded_in_question():
 builder=module('build_spider_executable_edits');rows=builder.candidates("select * from singer where age > 'terminal'",['*','age'],question='Which singers are older than 20?')
 assert any("age > 20" in row['sql'] for row in rows)

def test_string_literal_candidate_preserves_sql_quoting():
 builder=module('build_spider_executable_edits');rows=builder.candidates("select * from singer where country = 'terminal'",['*','country'],question='Which singers are from France?')
 assert any("country = 'France'" in row['sql'] for row in rows)

def test_database_group_fold_is_deterministic():
 selector=module('spider_advantage_selector_cv');assert selector.fold_of('concert_singer',5,2027)==selector.fold_of('concert_singer',5,2027)
 assert 0<=selector.fold_of('concert_singer',5,2027)<5

def test_selector_features_are_invariant_to_gold_and_labels():
 selector=module('spider_advantage_selector_cv')
 record={'question':'How many singers are older than 20?','predicted_sql':'select count(*) from singer where age > 20','gold_sql':'select secret','base_correct':False}
 candidate={'action':'replace_comparator','sql':'select count(*) from singer where age >= 20','executable':True,'correct':True,'advantage':1}
 stripped_record={key:value for key,value in record.items() if key not in {'gold_sql','base_correct'}}
 stripped_candidate={key:value for key,value in candidate.items() if key not in {'correct','advantage'}}
 assert selector.text(record,candidate)==selector.text(stripped_record,stripped_candidate)
