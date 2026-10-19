$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot
$dashboardPython = Join-Path $projectRoot ".venv-streamlit\Scripts\python.exe"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
$projectPython = if (Test-Path -LiteralPath $dashboardPython) { $dashboardPython } elseif (Test-Path -LiteralPath $venvPython) { $venvPython } else { "python" }

Push-Location $projectRoot
try {
    & $projectPython -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw 'Python 3.10+ is required. Create a compatible environment: py -3.13 -m venv .venv-streamlit'
    }
    & $projectPython -c "import importlib.util; import sys; sys.exit(0 if importlib.util.find_spec('streamlit') else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit is missing. Install dependencies: & '$projectPython' -m pip install -r requirements.txt"
    }
    & $projectPython -m streamlit run "app.py"
}
finally {
    Pop-Location
}
