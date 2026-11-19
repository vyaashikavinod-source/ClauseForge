import json

import pytest

from clauseforge.evaluation.portfolio import prepare, score
from clauseforge.training.config import load_config
from clauseforge.training.templates import render_prompt


def test_errors_and_invalid_labels_are_scored_not_dropped():
    rows = [
        {'target': 'a', 'prediction': 'a'},
        {'target': 'b', 'prediction': 'a'},
        {'target': 'b', 'prediction': None, 'error': 'TimeoutError'},
    ]
    metrics = score(rows, ['a', 'b', 'c'])
    assert metrics['accuracy'] == pytest.approx(1 / 3)
    assert metrics['per_category']['b']['support'] == 2
    assert metrics['per_category']['b']['accuracy_within_category'] == 0
    assert metrics['per_category']['c']['accuracy_within_category'] is None
    assert metrics['invalid_output_rate'] == pytest.approx(1 / 3)
    assert metrics['macro_f1_all_categories'] == pytest.approx(2 / 9)


def test_freezes_contract_disjoint_unique_sample_and_excludes_duplicate_text(tmp_path):
    from clauseforge.taxonomy import load_taxonomy_metadata
    taxonomy = load_taxonomy_metadata()
    data = tmp_path / 'data'; data.mkdir()
    (data / 'contracts.jsonl').write_text('\n'.join(json.dumps({'contract_id': c, 'raw_text': 'contract'}) for c in ['train', 'val', 'test']))
    (data / 'splits.json').write_text(json.dumps({'train': ['train'], 'validation': ['val'], 'test': ['test']}))
    rows = [{'clause_id': 'train', 'contract_id': 'train', 'text': 'duplicate', 'category': taxonomy[0].canonical},
            {'clause_id': 'test-duplicate', 'contract_id': 'test', 'text': 'DUPLICATE', 'category': taxonomy[0].canonical}]
    rows += [{'clause_id': str(i), 'contract_id': 'test', 'text': 'unique ' + str(i), 'category': taxonomy[i].canonical} for i in range(3)]
    (data / 'clauses.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    output = tmp_path / 'benchmark.json'
    assert prepare(data, output, count=3)['count'] == 3
    benchmark = json.loads(output.read_text())
    assert {e['contract_id'] for e in benchmark['examples']} == {'test'}
    assert all(e['text'].lower() != 'duplicate' for e in benchmark['examples'])
    with pytest.raises(ValueError, match='already exists'): prepare(data, output, count=3)


def test_shared_prompt_lists_taxonomy_for_a_real_zero_shot_baseline():
    prompt = render_prompt('A sample contract clause', 'cuad-classification-id-v3-taxonomy')
    assert 'governing_law:' in prompt
    assert 'A sample contract clause' in prompt
    assert 'untrusted' in prompt


def test_small_model_config_is_pinned_and_uses_benchmark_prompt():
    from pathlib import Path
    config = load_config(Path('training/configs/portfolio_qwen15b_qlora.yaml'))
    config.validate()
    assert config.model.name == 'Qwen/Qwen2.5-1.5B-Instruct'
    assert len(config.model.revision) == 40
    assert config.prompt_template_version == 'cuad-classification-id-v3-taxonomy'


def test_recorded_run_counts_provider_errors_and_keeps_unknown_cost_null(tmp_path, monkeypatch):
    import argparse
    from clauseforge.evaluation import portfolio
    benchmark = {'examples': [{'clause_id': '1', 'target': 'a', 'text': 'public sample'}, {'clause_id': '2', 'target': 'b', 'text': 'another sample'}],
                 'labels': [{'id': 'a'}, {'id': 'b'}], 'prompt_version': portfolio.PROMPT_VERSION}
    benchmark['benchmark_sha256'] = portfolio.digest(benchmark)
    source = tmp_path / 'benchmark.json'; source.write_text(json.dumps(benchmark))
    calls = []
    def provider(*args):
        calls.append(args)
        if len(calls) == 2: raise TimeoutError()
        return {'text': 'a', 'input_tokens': 10, 'output_tokens': 1, 'response_model': 'test'}
    monkeypatch.setattr(portfolio, 'call_model', provider)
    args = argparse.Namespace(benchmark=source, report=tmp_path / 'run.json', endpoint='http://127.0.0.1:8080/v1/chat/completions',
                              model='test', key_env='TEST_MISSING_KEY', max_requests=2, input_rate=None, output_rate=None, pricing_source=None, role='fine-tuned')
    report = portfolio.run(args)
    assert report['metrics']['sample_size'] == 2
    assert report['metrics']['accuracy'] == .5
    assert report['metrics']['request_error_rate'] == .5
    assert report['estimated_cost_usd'] is None
    assert report['input_tokens'] is None
    assert portfolio.compare([args.report], tmp_path / 'public.json')['runs'][0]['metrics']['accuracy'] == .5
    journal = args.report.with_suffix('.jsonl'); journal.write_text(journal.read_text().replace('TimeoutError', 'edited'))
    with pytest.raises(ValueError, match='checksum'): portfolio.compare([args.report], tmp_path / 'public2.json')


def test_quota_failure_stops_requests_and_cannot_be_compared(tmp_path, monkeypatch):
    import argparse
    import urllib.error
    from clauseforge.evaluation import portfolio
    benchmark = {'examples': [{'clause_id': str(i), 'target': 'a', 'text': 'public sample'} for i in range(3)],
                 'labels': [{'id': 'a'}], 'prompt_version': portfolio.PROMPT_VERSION}
    benchmark['benchmark_sha256'] = portfolio.digest(benchmark)
    source = tmp_path / 'benchmark.json'; source.write_text(json.dumps(benchmark))
    calls = []
    def provider(*args):
        calls.append(args)
        raise urllib.error.HTTPError('https://example.invalid', 429, 'quota', {}, None)
    monkeypatch.setattr(portfolio, 'call_model', provider)
    args = argparse.Namespace(benchmark=source, report=tmp_path / 'run.json', endpoint='http://127.0.0.1:8080/v1/chat/completions',
                              model='test', key_env='TEST_MISSING_KEY', max_requests=3, input_rate=0, output_rate=0,
                              pricing_source='free', role='api-zero-shot', request_interval=4.5, stop_on_http_error=True)
    report = portfolio.run(args)
    assert len(calls) == 1
    assert report['status'] == 'incomplete'
    assert report['planned_requests'] == 3
    assert json.loads(args.report.with_suffix('.jsonl').read_text())['http_status'] == 429
    with pytest.raises(ValueError, match='Incomplete'):
        portfolio.compare([args.report], tmp_path / 'comparison.json')
    original = args.report
    def recovered(*args):
        calls.append(args)
        return {'text': 'a', 'input_tokens': 10, 'output_tokens': 1, 'response_model': 'test'}
    monkeypatch.setattr(portfolio, 'call_model', recovered)
    monkeypatch.setattr(portfolio.time, 'sleep', lambda _: None)
    args.resume_from = original
    args.report = tmp_path / 'resumed.json'
    resumed = portfolio.run(args)
    assert len(calls) == 3  # The failed first clause was retained, never retried.
    assert resumed['status'] == 'complete'
    assert resumed['metrics']['accuracy'] == pytest.approx(2 / 3)
    assert resumed['resumed_from_records_sha256'] == report['records_sha256']
    assert original.with_suffix('.jsonl').read_bytes() == args.report.with_suffix('.jsonl').read_bytes()[:original.with_suffix('.jsonl').stat().st_size]
    args.request_interval = float('nan')
    with pytest.raises(ValueError, match='interval'):
        portfolio.run(args)
