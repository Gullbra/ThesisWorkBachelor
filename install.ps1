
function Invoke-Step {
  & $args
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

Invoke-Step py -3.12 -m venv venv
& .\venv\Scripts\Activate.ps1
Invoke-Step pip install -r requirements.txt
