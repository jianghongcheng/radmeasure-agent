"""Server-owned model connections. Jobs store an ID, never API credentials."""
import json
import os
import uuid
from pathlib import Path


def profiles():
    configured = json.loads(os.getenv('RADMEASURE_VLM_PROFILES') or '{}')
    if not isinstance(configured, dict) or 'default' in configured:
        raise ValueError('Invalid VLM profiles configuration')
    result = {'default': {'label': 'Default API', 'url': os.getenv('RADMEASURE_VLM_URL', ''),
                         'model': os.getenv('RADMEASURE_VLM_MODEL', ''),
                         'api_key': os.getenv('RADMEASURE_VLM_API_KEY', '')}}
    for name, label, prefix in [('openai', 'GPT (OpenAI)', 'OPENAI'), ('claude', 'Claude (Anthropic)', 'ANTHROPIC'), ('gemini', 'Gemini (Google)', 'GEMINI')]:
        result[name] = {'label': label, 'provider': name, 'url': '',
                        'model': os.getenv(prefix + '_MODEL', ''), 'api_key': os.getenv(prefix + '_API_KEY', '')}
    for name, value in configured.items():
        if not isinstance(value, dict) or not isinstance(value.get('url'), str) or not isinstance(value.get('model'), str):
            raise ValueError('Invalid VLM profile')
        result[name] = {**value, 'label': str(value.get('label', name))}
    return result


def public_profiles():
    values = profiles()
    names = ['openai', 'claude', 'gemini'] + [n for n in values if n not in {'openai', 'claude', 'gemini'}]
    return [{'id': name, 'label': values[name]['label'], 'model': values[name]['model'],
             'accepts_key': name in {'openai', 'claude', 'gemini'},
             'configured': bool(values[name]['model'] and (values[name].get('api_key') if values[name].get('provider') else values[name]['url']))}
            for name in names]


def credential_dir():
    return Path(os.getenv('RADMEASURE_PROVIDER_CREDENTIAL_DIR', str(Path(os.getenv('RADMEASURE_JOB_DB', 'runtime/jobs.db')).parent / 'provider-credentials')))


def resolve_profile(name):
    if name.startswith('request_'):
        token = name.removeprefix('request_')
        if len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
            raise KeyError('Unknown provider request')
        return json.loads((credential_dir() / (name + '.json')).read_text())
    return profiles()[name]


def save_request_profile(provider, model, api_key):
    if provider not in {'openai', 'claude', 'gemini'}:
        raise ValueError('Credentials are supported for GPT, Claude and Gemini')
    profile = profiles()[provider].copy()
    profile.update(model=model.strip() or profile['model'], api_key=api_key.strip() or profile['api_key'])
    if not profile['model'] or not profile['api_key']:
        raise ValueError('Enter a model ID and API key for the selected provider')
    folder = credential_dir()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    name = 'request_' + uuid.uuid4().hex
    fd = os.open(folder / (name + '.json'), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(profile, stream)
    return name
