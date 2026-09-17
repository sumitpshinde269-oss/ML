param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ScriptArgs
)

$venvPython = Join-Path $PSScriptRoot "ultralytics\.venv\Scripts\python.exe"
$mainScript = Join-Path $PSScriptRoot "main.py"

& $venvPython $mainScript @ScriptArgs
