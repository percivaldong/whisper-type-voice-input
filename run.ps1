$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
$env:HF_ENDPOINT = "https://hf-mirror.com"
& "$scriptDir\.venv\Scripts\python.exe" "$scriptDir\speak.py"
pause
