
function Invoke-Step {
  & $args[0] $args[1..($args.Count-1)]
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

Invoke-Step py -3.12 -m venv venv
& .\venv\Scripts\Activate.ps1
Invoke-Step pip install -r requirements.txt

$response = Read-Host "Would you like to download the BOSSbase dataset? (~1.6 GB) [y/N]"
if ($response -eq 'y' -or $response -eq 'Y') {
  $imagesDir = Join-Path $PSScriptRoot "images"
  $zipPath = Join-Path $imagesDir "BOSSbase_1.01.zip"

  New-Item -ItemType Directory -Force -Path $imagesDir | Out-Null

  Write-Host "Downloading dataset..."
  Invoke-WebRequest -Uri "https://dde.binghamton.edu/download/ImageDB/BOSSbase_1.01.zip" -OutFile $zipPath

  Write-Host "Extracting..."
  Expand-Archive -Path $zipPath -DestinationPath $imagesDir

  Remove-Item $zipPath
  Write-Host "Done! Dataset extracted to $imagesDir"

  & "$PSScriptRoot/images/categorizeImages.ps1" "$PSScriptRoot/images/BOSSbase_1.01" 15 15
}
