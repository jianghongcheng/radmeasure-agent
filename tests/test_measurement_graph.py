import copy
import hashlib
import io
import json
import math

import pytest

from radmeasure.measurement_graph import DiscrepancyPolicy, MeasurementGraph
from radmeasure.vision_measurement import VisionMeasurementClient, dual_path_registry


def landmark():
    def line(degrees):
        a = math.radians(degrees)
        return [[50, 50], [50 + 20*math.cos(a), 50 + 20*math.sin(a)]]
    return {'image_size': [256, 256], 'quality': {'passed': True},
            'axes': {'great_toe_axis': line(20), 'first_metatarsal_axis': line(0),
                     'second_metatarsal_axis': line(10)}, 'model': {'model_id': 'test-landmarks'}}


def estimate(hva, ima=10):
    return {'status': 'ok', 'angles': {'HVA': hva, 'IMA': ima},
            'evidence': {'HVA': 'toe and first metatarsal axes', 'IMA': 'first and second metatarsal axes'}}


class VLM:
    def __init__(self, values):
        self.values, self.feedback = values, []

    def estimate(self, artifact, question, protocols, feedback):
        self.feedback.append(copy.deepcopy(feedback))
        return copy.deepcopy(self.values[min(len(self.feedback)-1, len(self.values)-1)])


def run(values, *, geometry=None, repairs=1, protocols=('HVA', 'IMA')):
    vlm = VLM(values)
    workflow = MeasurementGraph(lambda a: geometry if geometry is not None else landmark(), vlm,
        DiscrepancyPolicy({'HVA': 1, 'IMA': 1}, {'HVA': 5, 'IMA': 5}, repairs),
        registry=dual_path_registry())
    return workflow.run({'path': 'test.png', 'sha256': 'fixture'}, 'Measure HVA and IMA', protocols), vlm


def test_keep_preserves_landmark_measurement_and_no_initial_leak():
    result, vlm = run([estimate(20.5)])
    assert result['routing']['decision'] == 'KEEP'
    assert result['measurements']['HVA'] == pytest.approx(20)
    assert vlm.feedback == [None]
    assert result['repair_attempts'] == 0


def test_repair_reestimates_only_vlm_then_rechecks():
    result, vlm = run([estimate(23), estimate(20.5)])
    assert result['routing']['decision'] == 'KEEP'
    assert result['repair_attempts'] == 1
    assert result['initial_vlm']['angles']['HVA'] == 23
    assert result['final_vlm']['angles']['HVA'] == 20.5
    assert result['measurements']['HVA'] == pytest.approx(20)
    assert vlm.feedback[1]['protocols_to_recheck'] == ['HVA']
    assert 'geometry_angles' not in vlm.feedback[1]
    assert [e['decision'] for e in result['agent_trajectory'] if e['step'] == 'controller'] == ['REPAIR', 'KEEP']


@pytest.mark.parametrize('values,repairs,reason,calls', [
    ([estimate(26)], 1, 'large_disagreement', 1),
    ([estimate(23)], 1, 'repair_budget_exhausted', 2),
    ([estimate(23)], 0, 'repair_budget_exhausted', 1),
    ([estimate(23), estimate(30)], 1, 'large_disagreement', 2),
    ([{'status': 'unresolved'}], 1, 'measurement_checks_failed', 1),
    ([estimate(float('nan'))], 1, 'measurement_checks_failed', 1),
    ([estimate(True)], 1, 'measurement_checks_failed', 1),
])
def test_stops_without_publishing_an_accepted_measurement(values, repairs, reason, calls):
    result, vlm = run(values, repairs=repairs)
    assert result['routing'] == {'decision': 'STOP', 'reason': reason}
    assert result['measurements'] == {}
    assert len(vlm.feedback) == calls


def test_invalid_geometry_overrides_angle_agreement():
    data = landmark()
    data['axes']['great_toe_axis'] = [[0, 0], [0, 0]]
    result, _ = run([estimate(20)], geometry=data)
    assert result['routing']['decision'] == 'STOP'


def test_invalid_protocol_does_not_call_models():
    result, vlm = run([estimate(20)], protocols=('UNKNOWN',))
    assert result['routing']['decision'] == 'STOP'
    assert vlm.feedback == []


def test_quality_failure_and_missing_evidence_stop():
    data = landmark(); data['quality']['passed'] = False
    assert run([estimate(20)], geometry=data)[0]['routing']['decision'] == 'STOP'
    value = estimate(20); value['evidence'] = {}
    assert run([value])[0]['routing']['decision'] == 'STOP'


def test_one_protocol_large_disagreement_stops_whole_case():
    result, _ = run([estimate(20, 30)])
    assert result['routing']['reason'] == 'large_disagreement'


def image_fixture():
    import numpy as np
    from PIL import Image
    stream = io.BytesIO()
    Image.fromarray(np.random.default_rng(7).integers(30, 220, (256, 256, 3), dtype=np.uint8)).save(stream, format='PNG')
    content = stream.getvalue()
    return content, {'path': 'test.png', 'sha256': hashlib.sha256(content).hexdigest(), 'media_type': 'image/png'}


def test_vlm_receives_real_image_and_only_registered_tools():
    content, artifact = image_fixture()
    payloads = []
    responses = [json.dumps({'tool': 'view_region', 'arguments': {'box': [.1, .1, .8, .8]}}), json.dumps(estimate(20))]
    def transport(payload):
        payloads.append(copy.deepcopy(payload))
        return responses.pop(0)
    client = VisionMeasurementClient('http://local/v1', 'test-vlm', transport=transport, loader=lambda p: content)
    result = client.estimate(artifact, 'Measure angles', dual_path_registry().describe())
    assert len(payloads) == 2
    assert payloads[0]['messages'][1]['content'][1]['image_url']['url'].startswith('data:image/png;base64,')
    assert result['tool_calls'][0]['tool'] == 'view_region'


@pytest.mark.parametrize('tool,args', [('shell', {'command': 'anything'}), ('view_region', {'box': [-1, 0, 1, 1]}), ('get_protocol', {'name': 'UNKNOWN'})])
def test_vlm_tool_arguments_cannot_escape_registry(tool, args):
    content, artifact = image_fixture()
    client = VisionMeasurementClient('http://local/v1', 'test', loader=lambda p: content,
        transport=lambda p: json.dumps({'tool': tool, 'arguments': args}))
    with pytest.raises(ValueError):
        client.estimate(artifact, 'Measure angles', dual_path_registry().describe())


def test_vlm_tool_budget_is_bounded():
    content, artifact = image_fixture()
    client = VisionMeasurementClient('http://local/v1', 'test', loader=lambda p: content,
        transport=lambda p: json.dumps({'tool': 'get_protocol', 'arguments': {'name': 'HVA'}}))
    with pytest.raises(ValueError, match='budget'):
        client.estimate(artifact, 'Measure angles', dual_path_registry().describe())


def test_worker_persists_keep_and_stop_and_human_review(tmp_path, monkeypatch):
    from radmeasure.jobs import SqliteJobRepository
    from radmeasure.pipeline import JobPipeline
    from radmeasure.production import DemoService
    from radmeasure.tools import RadMeasureTools
    from radmeasure.worker import Worker
    monkeypatch.delenv('RADMEASURE_MEASUREMENT_WORKFLOW', raising=False)
    for i, angle in enumerate([20.5, 30]):
        workflow = MeasurementGraph(lambda a: landmark(), VLM([estimate(angle)]),
            DiscrepancyPolicy({'HVA': 1, 'IMA': 1}, {'HVA': 5, 'IMA': 5}), registry=dual_path_registry())
        repository = SqliteJobRepository(tmp_path / f'jobs{i}.db')
        job, _ = repository.submit('uploaded_radiograph', {'artifact': {'path': 'test.png', 'sha256': 'fixture'}}, f'job{i}')
        Worker(repository, JobPipeline(RadMeasureTools(DemoService()), measurement_workflow=workflow)).run_once()
        saved = repository.get(job.job_id)
        assert saved.status == ('completed' if i == 0 else 'needs_review')
        assert saved.result['execution_record']['decision'] == ('KEEP' if i == 0 else 'STOP')
        if i:
            repository.review(job.job_id, 'reviewer', 'approve', {'HVA': 20.0, 'IMA': 10.0})
            assert repository.get(job.job_id).status == 'review_approved'


def test_invalid_numbers_preserve_evidence_as_strict_json():
    result, _ = run([estimate(float('nan'))])
    json.dumps(result, allow_nan=False)
    assert result['initial_vlm']['angles']['HVA'] == {'non_finite_number': 'nan'}


def test_single_requested_protocol():
    value = {'status': 'ok', 'angles': {'HVA': 20}, 'evidence': {'HVA': 'visible axes'}}
    result, _ = run([value], protocols=('HVA',))
    assert result['routing']['decision'] == 'KEEP'
    assert set(result['measurements']) == {'HVA'}


def test_model_failure_goes_to_review():
    class BrokenVLM:
        def estimate(self, *args):
            raise TimeoutError('unavailable')
    workflow = MeasurementGraph(lambda a: landmark(), BrokenVLM(),
        DiscrepancyPolicy({'HVA': 1}, {'HVA': 5}), registry=dual_path_registry())
    result = workflow.run({'path': 'test.png', 'sha256': 'fixture'}, 'HVA', ['HVA'])
    assert result['routing']['decision'] == 'STOP'
    assert result['final_vlm']['error_type'] == 'TimeoutError'


def test_upload_request_and_single_protocol_review(tmp_path, monkeypatch):
    import asyncio
    import httpx
    from radmeasure.api import create_app
    from radmeasure.jobs import SqliteJobRepository
    monkeypatch.setenv('RADMEASURE_DEMO_MODE', '1')
    monkeypatch.delenv('RADMEASURE_EVAL_REPLAY', raising=False)
    monkeypatch.setenv('RADMEASURE_JOB_DB', str(tmp_path / 'jobs.db'))
    monkeypatch.setenv('RADMEASURE_ARTIFACT_ROOT', str(tmp_path / 'artifacts'))
    monkeypatch.setenv('RADMEASURE_API_KEYS', json.dumps({'admin': {'name': 'reviewer', 'role': 'admin'}}))
    app = create_app()
    content, _ = image_fixture()
    async def check():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
            headers = {'x-api-key': 'admin', 'idempotency-key': 'upload'}
            response = await client.post('/v1/uploads', headers=headers,
                files={'file': ('image.png', content, 'image/png')}, data={'question': 'Measure HVA', 'protocols': 'HVA'})
            assert response.status_code == 202, response.text
            job = response.json()['job']
            assert job['payload']['question'] == 'Measure HVA'
            assert job['payload']['protocols'] == ['HVA']
            repository = SqliteJobRepository(tmp_path / 'jobs.db')
            repository.claim_next('test')
            repository.finish(job['job_id'], 'needs_review', {'measurements': {}, 'provenance': {'mode': 'dual_path_langgraph'}})
            endpoint = '/v1/jobs/' + job['job_id'] + '/review'
            rejected = await client.post(endpoint, headers=headers, json={'decision': 'approve'})
            assert rejected.status_code == 422
            reviewed = await client.post(endpoint, headers=headers,
                json={'decision': 'approve', 'corrected_measurements': {'HVA': 20}})
            assert reviewed.status_code == 200, reviewed.text
            assert reviewed.json()['result']['measurements'] == {'HVA': 20}
    asyncio.run(check())


def test_environment_factory_selects_landmark_model(tmp_path, monkeypatch):
    import radmeasure.vision_measurement as module
    monkeypatch.setenv('RADMEASURE_VLM_URL', 'http://model/v1')
    monkeypatch.setenv('RADMEASURE_VLM_MODEL', 'vision-test')
    monkeypatch.setenv('RADMEASURE_DISCREPANCY_POLICY', json.dumps({
        'keep_degrees': {'HVA': 1, 'IMA': 1}, 'stop_degrees': {'HVA': 5, 'IMA': 5}, 'max_repairs': 1}))
    monkeypatch.setattr(module, 'VisionMeasurementClient', lambda *args: VLM([estimate(20)]))
    class Inference:
        def predict_artifact(self, image_id, path, media_type, model_id):
            assert (image_id, path, model_id) == ('fixture', 'test.png', 'hvangle-landmarks')
            return landmark()
    workflow = module.graph_from_env(Inference())
    result = workflow.run({'sha256': 'fixture', 'path': 'test.png'}, 'Measure angles', ['HVA', 'IMA'])
    assert result['routing']['decision'] == 'KEEP'


def test_invalid_environment_policy_is_reviewable(tmp_path, monkeypatch):
    from radmeasure.jobs import SqliteJobRepository
    from radmeasure.pipeline import JobPipeline
    from radmeasure.production import DemoService
    from radmeasure.tools import RadMeasureTools
    from radmeasure.worker import Worker
    monkeypatch.delenv('RADMEASURE_MEASUREMENT_WORKFLOW', raising=False)
    monkeypatch.setenv('RADMEASURE_VLM_URL', 'http://model/v1')
    monkeypatch.setenv('RADMEASURE_VLM_MODEL', 'vision-test')
    monkeypatch.setenv('RADMEASURE_DISCREPANCY_POLICY', 'invalid-json')
    repository = SqliteJobRepository(tmp_path / 'jobs.db')
    job, _ = repository.submit('uploaded_radiograph', {'artifact': {'sha256': 'fixture', 'path': 'test.png'}}, 'bad-policy')
    Worker(repository, JobPipeline(RadMeasureTools(DemoService()), inference_client=object())).run_once()
    result = repository.get(job.job_id)
    assert result.status == 'needs_review'
    assert result.result['execution_record']['decision'] == 'STOP'


def test_api_profiles_select_model_without_exposing_credentials(monkeypatch):
    import radmeasure.vision_measurement as module
    from radmeasure.model_profiles import public_profiles
    monkeypatch.setenv('RADMEASURE_VLM_PROFILES', json.dumps({'local': {
        'label': 'Local vision API', 'url': 'http://vision/v1', 'model': 'vision-model', 'api_key': 'test-private-key'}}))
    monkeypatch.setenv('RADMEASURE_DISCREPANCY_POLICY', json.dumps({
        'keep_degrees': {'HVA': 1, 'IMA': 1}, 'stop_degrees': {'HVA': 5, 'IMA': 5}}))
    public = public_profiles()
    assert public[-1]['id'] == 'local'
    assert public[-1]['configured'] is True
    assert 'test-private-key' not in json.dumps(public)
    called = []
    monkeypatch.setattr(module, 'VisionMeasurementClient', lambda *args: called.append(args) or VLM([estimate(20)]))
    module.graph_from_env(object(), 'local')
    assert called == [('http://vision/v1', 'vision-model', 'test-private-key')]
    with pytest.raises(KeyError):
        module.graph_from_env(object(), 'unknown')
