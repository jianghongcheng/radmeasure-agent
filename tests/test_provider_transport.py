import json
import pytest
from radmeasure.provider_transport import ProviderTransport
from radmeasure.model_profiles import public_profiles, save_request_profile, resolve_profile

PAYLOAD = {'messages': [{'role': 'system', 'content': 'Return JSON'},
    {'role': 'user', 'content': [{'type': 'text', 'text': 'Measure HVA'},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,aGVsbG8='}}]},
    {'role': 'assistant', 'content': '{"tool":"get_protocol"}'},
    {'role': 'user', 'content': [{'type': 'text', 'text': 'Protocol definition'}]}]}

@pytest.mark.parametrize('provider,host', [('openai','api.openai.com'),('claude','api.anthropic.com'),('gemini','generativelanguage.googleapis.com')])
def test_native_image_requests(provider, host, monkeypatch):
    adapter = ProviderTransport(provider, 'model-id', 'test-secret')
    req = adapter.request(PAYLOAD)
    body = json.loads(req.data)
    assert host in req.full_url
    assert 'test-secret' not in req.full_url and 'test-secret' not in req.data.decode()
    if provider == 'openai':
        assert body['input'][0]['content'][1]['type'] == 'input_image'
        assert body['input'][1]['content'][0]['type'] == 'output_text'
        response = {'status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{"status":"ok"}'}]}]}
    elif provider == 'claude':
        assert body['messages'][0]['content'][1]['source']['media_type'] == 'image/png'
        assert req.get_header('Anthropic-version') == '2023-06-01'
        response = {'stop_reason':'end_turn','content':[{'type':'text','text':'```json\n{"status":"ok"}\n```'}]}
    else:
        assert body['contents'][0]['parts'][1]['inlineData']['data'] == 'aGVsbG8='
        assert body['contents'][1]['role'] == 'model'
        response = {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"status":"ok"}'}]}}]}
    assert json.loads(adapter.decode(response)) == {'status':'ok'}
    import io
    import urllib.request
    requests=[]
    def fake_urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(json.dumps(response).encode())
    monkeypatch.setattr(urllib.request,'urlopen',fake_urlopen)
    assert json.loads(adapter(PAYLOAD)) == {'status':'ok'}
    assert host in requests[0].full_url

@pytest.mark.parametrize('provider,response', [('openai',{'status':'incomplete'}),('claude',{'stop_reason':'max_tokens'}),('gemini',{'candidates':[]})])
def test_no_partial_or_blocked_result(provider,response):
    with pytest.raises(ValueError):
        ProviderTransport(provider,'model','key').decode(response)


def test_private_job_credentials(tmp_path,monkeypatch):
    monkeypatch.setenv('RADMEASURE_PROVIDER_CREDENTIAL_DIR',str(tmp_path))
    name=save_request_profile('claude','my-vision-model','private-test-key')
    assert 'private-test-key' not in name
    assert resolve_profile(name)['api_key']=='private-test-key'
    assert (tmp_path/(name+'.json')).stat().st_mode & 0o777 == 0o600
    public=public_profiles()
    assert [p['id'] for p in public[:3]]==['openai','claude','gemini']
    assert 'private-test-key' not in json.dumps(public)
    with pytest.raises(KeyError):resolve_profile('request_../../secret')
    with pytest.raises(ValueError):save_request_profile('arbitrary','model','secret')


def test_upload_accepts_provider_model_key_without_leaking_key(tmp_path,monkeypatch):
    import asyncio
    import io
    import httpx
    from PIL import Image
    from radmeasure.api import create_app
    monkeypatch.setenv('RADMEASURE_DEMO_MODE','1')
    monkeypatch.setenv('RADMEASURE_EVAL_REPLAY','0')
    monkeypatch.setenv('RADMEASURE_JOB_DB',str(tmp_path/'jobs.db'))
    monkeypatch.setenv('RADMEASURE_ARTIFACT_ROOT',str(tmp_path/'images'))
    monkeypatch.setenv('RADMEASURE_PROVIDER_CREDENTIAL_DIR',str(tmp_path/'keys'))
    monkeypatch.setenv('RADMEASURE_API_KEYS',json.dumps({'local':{'name':'test','role':'operator'}}))
    app=create_app()
    image=io.BytesIO();Image.new('RGB',(256,256)).save(image,format='PNG')
    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
            for provider in ('openai','claude','gemini'):
                result=await client.post('/v1/uploads',headers={'x-api-key':'local','idempotency-key':provider},
                    files={'file':('image.png',image.getvalue(),'image/png')},
                    data={'api_profile':provider,'model':'test-vision','provider_key':'mock-private-key'})
                assert result.status_code==202,result.text
                assert 'mock-private-key' not in result.text
                profile=resolve_profile(result.json()['job']['payload']['api_profile'])
                assert profile['provider']==provider
                assert profile['model']=='test-vision'
                assert profile['api_key']=='mock-private-key'
    asyncio.run(run())
