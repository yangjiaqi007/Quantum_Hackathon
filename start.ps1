param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Set up the Python environment first using README.md.'
}
# Local-only service, one worker owns the persistent queue.
& $projectPython -m uvicorn backend.main:app --host 127.0.0.1 --port $Port --workers 1
exit $LASTEXITCODE
