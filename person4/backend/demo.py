"""Submit/cache any approved demo request against an already running server."""
import argparse
import json
from pathlib import Path
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def exchange(url, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(url, data=data, headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('request_file', type=Path)
    parser.add_argument('--server', default='http://127.0.0.1:8000')
    parser.add_argument('--fresh', action='store_true', help='Bypass existing cache and save a fresh result')
    parser.add_argument('--wait', type=int, default=360, help='Maximum polling seconds')
    parser.add_argument('--output', type=Path, help='Optional file for the complete result JSON')
    args = parser.parse_args()
    from .models import HarmonizeRequest
    payload = HarmonizeRequest.model_validate_json(args.request_file.read_text(encoding='utf-8'))
    if args.fresh:
        payload.use_cache = False
    try:
        submitted = exchange(args.server.rstrip('/') + '/jobs', payload.model_dump(mode='json'))
        job_id = submitted['job_id']
        print('Job:', job_id)
        end = time.monotonic() + args.wait
        previous = None
        while time.monotonic() < end:
            job = exchange(f'{args.server}/jobs/{job_id}')
            stage = job['progress']['stage']
            if stage != previous:
                print(stage)
                previous = stage
            if job['status'] == 'completed':
                result = exchange(f'{args.server}/jobs/{job_id}/result')
                print('Execution:', result['result']['execution']['mode'])
                print('Saved result reused:', result['provenance']['cache_hit'])
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
                    print('Result saved:', args.output)
                return 0
            if job['status'] in ('failed', 'cancelled'):
                print('Job stopped:', job.get('error') or job['status'])
                return 1
            time.sleep(.5)
        print('Stopped waiting; the server job may still be running. Check:', job_id)
        return 1
    except HTTPError as exc:
        print('API error:', exc.code, exc.read().decode())
        return 1
    except URLError:
        print('Cannot reach the backend. Start it first with start.ps1.')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
