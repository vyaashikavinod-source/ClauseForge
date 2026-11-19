from pathlib import Path

import pytest

from clauseforge.serving.errors import ProviderUnavailableError
from clauseforge.serving.providers.vllm import VllmProvider


def provider(url):
    return VllmProvider(Path('model-id'), url, (), 32, 0, 5)


def test_remote_key_is_not_sent_over_plain_http(monkeypatch):
    monkeypatch.setenv('CLAUSEFORGE_REMOTE_API_KEY', 'test-only-not-secret')
    with pytest.raises(ProviderUnavailableError):
        provider('http://127.0.0.1:8080')._headers()


def test_remote_key_is_sent_only_in_https_header(monkeypatch):
    monkeypatch.setenv('CLAUSEFORGE_REMOTE_API_KEY', 'test-only-not-secret')
    headers = provider('https://model.example')._headers()
    assert headers['Authorization'] == 'Bearer test-only-not-secret'
    monkeypatch.delenv('CLAUSEFORGE_REMOTE_API_KEY')
    assert 'Authorization' not in provider('http://127.0.0.1:8080')._headers()
