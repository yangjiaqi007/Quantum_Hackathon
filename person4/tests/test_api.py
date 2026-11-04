import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from person4.backend.config import Settings
from person4.backend.main import create_app
from person4.backend.storage import Store, QueueFull, cache_key


def request_data(**changes):
    data = {'melody': {'notes': [{'midi': 64, 'duration': 2}, {'midi': 65, 'duration': 1}]}}
    data.update(changes)
    return data


def wait_for_job(client, job_id, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f'/jobs/{job_id}')
        assert response.status_code == 200
        job = response.json()
        if job['status'] in ('completed', 'failed', 'cancelled'):
            return job
        time.sleep(.05)
    pytest.fail('Job did not finish in time')


def slow_runner(request, settings, progress):
    progress({'stage': 'Test sleeping', 'completed_windows': 0, 'total_windows': 1})
    time.sleep(60)
    return {}


def broken_runner(request, settings, progress):
    raise RuntimeError('private-test-secret')


def test_mock_submission_result_cache_and_restart(tmp_path):
    settings = Settings(data_dir=tmp_path)
    with TestClient(create_app(settings)) as client:
        assert client.get('/health').status_code == 200
        assert client.get('/capabilities').json()['configured_backends'] == ['mock']
        submitted = client.post('/jobs', json=request_data())
        assert submitted.status_code == 202
        job_id = submitted.json()['job_id']
        assert submitted.headers['location'] == f'/jobs/{job_id}'
        finished = wait_for_job(client, job_id)
        assert finished['status'] == 'completed', finished
        result = client.get(f'/jobs/{job_id}/result').json()
        assert result['result']['execution']['is_mock'] is True
        assert result['result']['voice_order'] == ['bass', 'tenor', 'alto', 'soprano']
        assert result['result']['voices']['soprano'] == [
            {'midi': 64, 'onset': 0., 'duration': 2.}, {'midi': 65, 'onset': 2., 'duration': 1.}]
        assert result['result']['quantum_report']['max_qubits_per_window'] is None
        second = client.post('/jobs', json=request_data()).json()
        assert second['status'] == 'completed'
        assert second['provenance']['cache_hit'] is True
        assert second['provenance']['source_job_id'] == job_id
        assert client.post(f'/jobs/{job_id}/cancel').status_code == 409
    with TestClient(create_app(settings)) as client:
        assert client.get(f'/jobs/{job_id}/result').json()['result'] == result['result']
        fresh = client.post('/jobs', json=request_data(use_cache=False)).json()
        assert fresh['provenance']['cache_hit'] is False
        assert wait_for_job(client, fresh['job_id'])['status'] == 'completed'


@pytest.mark.parametrize('payload', [
    {'melody': {'notes': []}},
    {'melody': {'notes': [{'midi': 200, 'duration': 1}]}},
    {'melody': {'notes': [{'midi': 64, 'duration': 0}]}},
    {'melody': {'notes': [{'midi': True, 'duration': 1}]}},
    request_data(run={'backend': 'ibm'}),
    request_data(run={'backend': 'aer'}),
    request_data(ibm_token='not-accepted'),
])
def test_bad_requests_rejected(tmp_path, payload):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.post('/jobs', json=payload).status_code == 422


def test_unknown_job_and_cors(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        assert client.get('/jobs/not-a-job').status_code == 404
        assert client.get('/jobs/00000000-0000-0000-0000-000000000000').status_code == 404
        response = client.options('/jobs', headers={
            'Origin': 'http://localhost:5173', 'Access-Control-Request-Method': 'POST'})
        assert response.headers['access-control-allow-origin'] == 'http://localhost:5173'
        bad = client.options('/jobs', headers={
            'Origin': 'https://example.com', 'Access-Control-Request-Method': 'POST'})
        assert 'access-control-allow-origin' not in bad.headers


def test_failure_message_is_safe(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path), runner=broken_runner)) as client:
        job_id = client.post('/jobs', json=request_data()).json()['job_id']
        finished = wait_for_job(client, job_id)
        assert finished['status'] == 'failed'
        assert 'private-test-secret' not in str(finished)
        assert client.get(f'/jobs/{job_id}/result').status_code == 409


def test_cancellation_and_queue_limit(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, max_pending_jobs=1), runner=slow_runner)) as client:
        job_id = client.post('/jobs', json=request_data()).json()['job_id']
        assert client.get(f'/jobs/{job_id}/result').status_code == 409
        assert client.post('/jobs', json=request_data(use_cache=False)).status_code == 429
        cancelled = client.post(f'/jobs/{job_id}/cancel').json()
        assert cancelled['status'] == 'cancelled'
        assert client.get(f'/jobs/{job_id}/result').status_code == 409


def test_timeout_kills_local_work(tmp_path):
    with TestClient(create_app(Settings(data_dir=tmp_path, job_timeout_seconds=2), runner=slow_runner)) as client:
        job_id = client.post('/jobs', json=request_data()).json()['job_id']
        finished = wait_for_job(client, job_id)
        assert finished['status'] == 'failed'
        assert finished['error']['code'] == 'timeout'


def test_restart_marks_running_failed_and_preserves_queued(tmp_path):
    from person4.backend.models import HarmonizeRequest
    settings = Settings(data_dir=tmp_path)
    store = Store(tmp_path)
    payload = HarmonizeRequest.model_validate(request_data()).model_dump(mode='json')
    running = store.create(payload, settings.cache_namespace, 20)
    store.claim()
    queued = store.create(payload, settings.cache_namespace, 20)
    with TestClient(create_app(settings)) as client:
        failed = client.get(f"/jobs/{running['id']}").json()
        assert failed['error']['code'] == 'interrupted'
        assert wait_for_job(client, queued['id'])['status'] == 'completed'


def test_cache_separates_settings_and_run_options():
    one = request_data()
    two = request_data(run={'backend': 'aer'})
    assert cache_key(one, 'v1') != cache_key(two, 'v1')
    assert cache_key(one, 'v1') != cache_key(one, 'v2')
    assert cache_key(one, 'v1') == cache_key(dict(one, use_cache=False), 'v1')


def test_queued_jobs_do_not_run_under_changed_configuration(tmp_path):
    from person4.backend.models import HarmonizeRequest
    store = Store(tmp_path)
    payload = HarmonizeRequest.model_validate(request_data()).model_dump(mode='json')
    old = store.create(payload, 'old-version', 20)
    with TestClient(create_app(Settings(data_dir=tmp_path))) as client:
        job = client.get(f"/jobs/{old['id']}").json()
        assert job['status'] == 'failed'
        assert job['error']['code'] == 'configuration_changed'


def test_saved_hardware_demo_can_be_loaded_without_credentials(tmp_path):
    from person4.backend.models import HarmonizeRequest
    settings = Settings(data_dir=tmp_path)
    store = Store(tmp_path)
    payload = HarmonizeRequest.model_validate(request_data(run={'backend': 'ibm'})).model_dump(mode='json')
    original = store.create(payload, settings.cache_namespace, 20)
    store.claim()
    # Only a persistence fixture; no hardware execution or result is invented.
    store.complete(original['id'], {'fixture': 'cached-provider-result'})
    with TestClient(create_app(settings)) as client:
        cached = client.post('/jobs', json=payload).json()
        assert cached['status'] == 'completed'
        assert cached['provenance']['cache_hit'] is True
        assert cached['provenance']['execution'] == 'ibm'
        payload['use_cache'] = False
        assert client.post('/jobs', json=payload).status_code == 422
