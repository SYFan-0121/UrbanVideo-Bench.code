<#
.SYNOPSIS
筛选 Hugging Face 未提供的本地视频并通过 SSH 上传。
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$ServerIp,
    [string]$ServerUser = 'syf',
    [string]$VideoDirectory = (Join-Path (Split-Path $PSScriptRoot -Parent) 'dataset\videos'),
    [string]$HubRevision = 'main'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $VideoDirectory -PathType Container)) {
    throw "Local video directory not found: $VideoDirectory"
}

$treeUrl = "https://huggingface.co/api/datasets/EmbodiedCity/UrbanVideo-Bench/tree/$HubRevision?recursive=true&expand=false"
$hubFiles = Invoke-RestMethod -Uri $treeUrl -TimeoutSec 60
$hubVideoNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($entry in $hubFiles) {
    if ($entry.type -eq 'file' -and $entry.path -like 'videos/*.mp4') {
        [void]$hubVideoNames.Add([System.IO.Path]::GetFileName($entry.path))
    }
}

$localVideos = @(Get-ChildItem -LiteralPath $VideoDirectory -File -Filter '*.mp4')
$supplement = @($localVideos | Where-Object { -not $hubVideoNames.Contains($_.Name) })
$totalBytes = ($supplement | Measure-Object -Property Length -Sum).Sum
Write-Host "Local videos: $($localVideos.Count)"
Write-Host "Not present in Hugging Face ${HubRevision}: $($supplement.Count)"
Write-Host "Upload size: $([math]::Round(($totalBytes / 1GB), 2)) GiB"

if ($supplement.Count -eq 0) {
    Write-Host 'No supplemental videos to upload.'
    exit 0
}

$destination = "${ServerUser}@${ServerIp}:D:/VLM/syf/incoming-local-videos/"
$scpArguments = @($supplement | ForEach-Object { $_.FullName }) + @($destination)
Write-Host "Uploading only local MP4 names absent from the selected Hub revision to $destination"
& scp @scpArguments
if ($LASTEXITCODE -ne 0) {
    throw "scp failed with exit code $LASTEXITCODE"
}
Write-Host 'Upload complete. Run merge_supplemental_videos.py on the server to add these without overwriting dataset files.'
