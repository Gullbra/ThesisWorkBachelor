param(
  [Parameter(Mandatory)][string]$FolderPath,
  [Parameter(Mandatory)][int]$ValPercent,
  [Parameter(Mandatory)][int]$TestPercent
)

Write-Host $FolderPath

$trainPct = 100 - $ValPercent - $TestPercent
if ($trainPct -le 0) {
  Write-Error "val + test percentages must be less than 100"
  exit 1
}

if ((Get-ChildItem $FolderPath -File).Count -eq 0) {
  Write-Error "Dataset folder empty"
  Write-Error "Destination folders are not empty, aborting to prevent data loss."
  exit 1
}

$existingFiles = Get-ChildItem "$FolderPath\train\cover\", "$FolderPath\val\cover\", "$FolderPath\test\cover\" -ErrorAction SilentlyContinue
if ($existingFiles.Count -gt 0) {
  Write-Error "Destination folders are not empty, aborting to prevent data loss."
  exit 1
}

Write-Host "Split: train=$trainPct% val=$ValPercent% test=$TestPercent%"

$response = Read-Host "Would you like to convert images to 256x256 .png [y/N]?"
if ($response -eq 'y' -or $response -eq 'Y') {
  python "$PSScriptRoot/convert_to_png.py" $FolderPath
}

$images = @(Get-ChildItem -Path $FolderPath -Filter "*.pgm" -File) +
          @(Get-ChildItem -Path $FolderPath -Filter "*.png" -File) +
          @(Get-ChildItem -Path $FolderPath -Filter "*.jpg"  -File) +
          @(Get-ChildItem -Path $FolderPath -Filter "*.jpeg" -File)
$total = $images.Count

$nTest  = [math]::Floor($total * $TestPercent / 100)
$nVal   = [math]::Floor($total * $ValPercent  / 100)
$nTrain = $total - $nTest - $nVal

Write-Host "Total: $total images → train=$nTrain val=$nVal test=$nTest"

New-Item -ItemType Directory -Force -Path "$FolderPath\train\cover", "$FolderPath\val\cover", "$FolderPath\test\cover" | Out-Null

Write-Host "Moving images..."

# Build index ranges
$trainImages = $images[0..($nTrain - 1)]
$valImages   = $images[$nTrain..($nTrain + $nVal - 1)]
$testImages  = $images[($nTrain + $nVal)..($total - 1)]

# Move in bulk
$trainImages | Move-Item -Destination "$FolderPath\train\cover\"
$valImages   | Move-Item -Destination "$FolderPath\val\cover\"
$testImages  | Move-Item -Destination "$FolderPath\test\cover\"

# Renumber or number
python "$PSScriptRoot/renumber_images.py" "$FolderPath\train\cover\"
python "$PSScriptRoot/renumber_images.py" "$FolderPath\val\cover\"
python "$PSScriptRoot/renumber_images.py" "$FolderPath\test\cover\"

Write-Host "Done!"
