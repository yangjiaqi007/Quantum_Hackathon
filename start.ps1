param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
& (Join-Path $PSScriptRoot 'person4\start.ps1') -Port $Port
exit $LASTEXITCODE
