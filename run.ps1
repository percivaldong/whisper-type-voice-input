$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
& "$scriptDir\.venv\Scripts\python.exe" "$scriptDir\speak.py"
pause
