"""Server-only configuration. Credentials never enter a request or a response."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: ROOT / '.local-data')
    cors_origins: list[str] = field(default_factory=lambda: [
        'http://localhost:3000', 'http://localhost:5173',
        'http://127.0.0.1:3000', 'http://127.0.0.1:5173'])
    max_pending_jobs: int = 20
    job_timeout_seconds: int = 300
    pipeline_version: str = 'v1'
    pipeline_mode: str = 'mock'
    compiler_callable: str = ''
    solver_callable: str = ''
    checker_callable: str = ''
    ibm_enabled: bool = False
    ibm_token: str | None = field(default=None, repr=False)
    ibm_instance: str | None = None
    ibm_backend: str | None = None
    fake_backend: str = 'FakeGuadalupeV2'

    def __post_init__(self):
        self.data_dir = Path(self.data_dir).resolve()
        if self.pipeline_mode not in ('mock', 'team'):
            raise ValueError('PIPELINE_MODE must be mock or team')
        if self.max_pending_jobs < 1 or self.job_timeout_seconds < 1:
            raise ValueError('Job limits must be positive')

    @property
    def cache_namespace(self) -> str:
        # Increment PIPELINE_VERSION whenever teammate code or rule weights change.
        config = {k: getattr(self, k) for k in (
            'pipeline_version', 'pipeline_mode', 'compiler_callable',
            'solver_callable', 'checker_callable', 'ibm_instance', 'ibm_backend', 'fake_backend')}
        return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / '.env', override=False)
        value = cls()
        for name in ('pipeline_version', 'pipeline_mode', 'compiler_callable',
                     'solver_callable', 'checker_callable', 'fake_backend'):
            setattr(value, name, os.getenv(name.upper(), getattr(value, name)))
        for name in ('ibm_token', 'ibm_instance', 'ibm_backend'):
            setattr(value, name, os.getenv(name.upper()) or None)
        value.ibm_enabled = os.getenv('IBM_ENABLED', 'false').lower() == 'true'
        value.data_dir = Path(os.getenv('DATA_DIR', str(value.data_dir)))
        if not value.data_dir.is_absolute():
            value.data_dir = ROOT / value.data_dir
        if os.getenv('CORS_ORIGINS'):
            value.cors_origins = [x.strip() for x in os.environ['CORS_ORIGINS'].split(',') if x.strip()]
        value.max_pending_jobs = int(os.getenv('MAX_PENDING_JOBS', '20'))
        value.job_timeout_seconds = int(os.getenv('JOB_TIMEOUT_SECONDS', '300'))
        value.__post_init__()
        return value
