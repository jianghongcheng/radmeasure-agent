import importlib.util
from pathlib import Path

PATH=Path(__file__).resolve().parents[1]/"scripts"/"spider_execution_only_baseline.py"
SPEC=importlib.util.spec_from_file_location("execution_only",PATH);MODULE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(MODULE)

def test_execution_only_never_edits_valid_base():
    records=[{"index":0,"db_id":"d","base_executable":True,"base_correct":True,"candidates":[{"action":"x","executable":True,"correct":False}]}]
    result=MODULE.evaluate(records)
    assert result["harm_count"]==0
    assert result["coverage"]==0
    assert result["database_clustered_95pct_ci"]==[0.0,0.0]

def test_execution_only_uses_first_executable_candidate_for_invalid_base():
    records=[{"index":0,"db_id":"d","base_executable":False,"base_correct":False,"candidates":[
        {"action":"bad","executable":False,"correct":False},{"action":"repair","executable":True,"correct":True},{"action":"later","executable":True,"correct":False}]}]
    result=MODULE.evaluate(records)
    assert result["benefit_count"]==1
    assert result["cases"][0]["chosen_action"]=="repair"

def test_cascade_prioritizes_structural_execution_repair():
    execution={"cases":[{"index":0,"db_id":"d","before_correct":False,"after_correct":True,"edited":True},
                        {"index":1,"db_id":"d","before_correct":False,"after_correct":False,"edited":False}]}
    selector={"folds":[{"methods":{"learned_selector_learned_proposal":{"cases":[
        {"index":0,"before_correct":False,"after_correct":False,"edited":True},
        {"index":1,"before_correct":False,"after_correct":True,"edited":True}]}}}]}
    result=MODULE.combine_with_selector(execution,selector)
    assert result["benefit_count"]==2
    assert result["execution_stage_count"]==1
