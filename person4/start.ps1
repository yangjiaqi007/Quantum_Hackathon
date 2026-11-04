param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
$projectPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Set up the Python environment first using README.md.'
}
# Local-only service, one worker owns the persistent queue.
& $projectPython -m uvicorn person4.backend.main:app --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
