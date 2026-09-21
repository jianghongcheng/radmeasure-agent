"""Translate image messages to each provider's native REST format."""
import json
import urllib.request
from urllib.parse import quote


class ProviderTransport:
    def __init__(self, provider, model, api_key, timeout=60):
        self.provider, self.model, self.key, self.timeout = provider, model, api_key, timeout

    def request(self, payload):
        system = '\n'.join(m['content'] for m in payload['messages'] if m['role'] == 'system')
        messages = []
        for m in payload['messages']:
            if m['role'] == 'system':
                continue
            content = m['content']
            parts = [{'type': 'text', 'text': content}] if isinstance(content, str) else content
            converted = []
            for p in parts:
                if p['type'] == 'text':
                    if self.provider == 'openai':
                        converted.append({'type': 'output_text' if m['role'] == 'assistant' else 'input_text', 'text': p['text']})
                    elif self.provider == 'gemini':
                        converted.append({'text': p['text']})
                    else:
                        converted.append(p)
                else:
                    uri = p['image_url']['url']
                    meta, data = uri.split(',', 1)
                    mime = meta.split(':', 1)[1].split(';')[0]
                    if self.provider == 'openai':
                        converted.append({'type': 'input_image', 'image_url': uri})
                    elif self.provider == 'claude':
                        converted.append({'type': 'image', 'source': {'type': 'base64', 'media_type': mime, 'data': data}})
                    else:
                        converted.append({'inlineData': {'mimeType': mime, 'data': data}})
            if self.provider == 'gemini':
                messages.append({'role': 'model' if m['role'] == 'assistant' else 'user', 'parts': converted})
            else:
                messages.append({'role': m['role'], 'content': converted})
        headers = {'Content-Type': 'application/json'}
        if self.provider == 'openai':
            url = 'https://api.openai.com/v1/responses'
            headers['Authorization'] = 'Bearer ' + self.key
            body = {'model': self.model, 'instructions': system, 'input': messages,
                    'max_output_tokens': 4096, 'store': False, 'text': {'format': {'type': 'json_object'}}}
        elif self.provider == 'claude':
            url = 'https://api.anthropic.com/v1/messages'
            headers.update({'x-api-key': self.key, 'anthropic-version': '2023-06-01'})
            body = {'model': self.model, 'system': system, 'messages': messages, 'max_tokens': 4096}
        elif self.provider == 'gemini':
            url = 'https://generativelanguage.googleapis.com/v1beta/models/' + quote(self.model, safe='') + ':generateContent'
            headers['x-goog-api-key'] = self.key
            body = {'systemInstruction': {'parts': [{'text': system}]}, 'contents': messages,
                    'generationConfig': {'responseMimeType': 'application/json', 'maxOutputTokens': 4096}}
        else:
            raise ValueError('Unknown image provider')
        return urllib.request.Request(url, data=json.dumps(body, allow_nan=False).encode(), headers=headers)

    def decode(self, value):
        if self.provider == 'openai':
            if value.get('status') != 'completed':
                raise ValueError('Provider response incomplete')
            text = ''.join(p.get('text', '') for item in value.get('output', []) if item.get('type') == 'message'
                           for p in item.get('content', []) if p.get('type') == 'output_text')
        elif self.provider == 'claude':
            if value.get('stop_reason') != 'end_turn':
                raise ValueError('Provider response incomplete')
            text = ''.join(p['text'] for p in value.get('content', []) if p.get('type') == 'text')
        else:
            candidates = value.get('candidates', [])
            if not candidates or candidates[0].get('finishReason') != 'STOP':
                raise ValueError('Provider response blocked or incomplete')
            text = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', []) if not p.get('thought'))
        text = text.strip()
        if text.startswith('```') and text.endswith('```'):
            text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        if not text:
            raise ValueError('Empty provider response')
        return text

    def __call__(self, payload):
        with urllib.request.urlopen(self.request(payload), timeout=self.timeout) as response:
            return self.decode(json.load(response))
