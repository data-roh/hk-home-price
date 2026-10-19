$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$projectPython = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

Push-Location $projectRoot
try {
    & $projectPython "build_monthly_master.py"
    if ($LASTEXITCODE -ne 0) { throw "Monthly master build failed." }
    & $projectPython "run_research_audit.py"
    if ($LASTEXITCODE -ne 0) { throw "Research audit failed." }
}
finally {
    Pop-Location
}
