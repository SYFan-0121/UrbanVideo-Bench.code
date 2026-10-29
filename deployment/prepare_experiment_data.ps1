param([string]$RepoRoot = (Split-Path -Parent $PSScriptRoot))

# NTFS 硬链接共享数据字节，兼容 SSH 账户禁止遍历目录联接的服务器策略。
$ErrorActionPreference = 'Stop'
$RepoRoot = [IO.Path]::GetFullPath($RepoRoot)
$source = Join-Path $RepoRoot 'dataset'
if (-not (Test-Path -LiteralPath (Join-Path $source 'MCQ.parquet'))) { throw 'Missing dataset/MCQ.parquet.' }
$videos = @(Get-ChildItem -LiteralPath (Join-Path $source 'videos') -Filter '*.mp4' -File)
$files = @((Get-Item -LiteralPath (Join-Path $source 'MCQ.parquet')))
if (Test-Path -LiteralPath (Join-Path $source 'download_manifest.json')) {
    $files += Get-Item -LiteralPath (Join-Path $source 'download_manifest.json')
}
$files += $videos

foreach ($name in @('agentic', 'uniform32')) {
    $destination = Join-Path (Join-Path $RepoRoot $name) 'dataset'
    $item = Get-Item -LiteralPath $destination -Force -ErrorAction SilentlyContinue
    if ($item -and $item.LinkType -eq 'Junction') {
        if ($item.Target.Count -ne 1 -or [IO.Path]::GetFullPath($item.Target[0]) -ne $source) { throw "Unexpected dataset junction: $destination" }
        # 只删除目录联接本身，不递归访问或删除源数据。
        [IO.Directory]::Delete($destination)
    }
    New-Item -ItemType Directory -Force (Join-Path $destination 'videos') | Out-Null
    foreach ($file in $files) {
        $relative = $file.FullName.Substring($source.Length + 1)
        $target = Join-Path $destination $relative
        if (Test-Path -LiteralPath $target) {
            $existing = Get-Item -LiteralPath $target
            if ($existing.LinkType -ne 'HardLink' -or $existing.Target -notcontains $file.FullName) { throw "Dataset entry is not the expected hard link: $target" }
        } else {
            New-Item -ItemType HardLink -Path $target -Target $file.FullName | Out-Null
        }
    }
    Write-Output "$name shares $($videos.Count) videos and $($files.Count - $videos.Count) metadata files via hard links."
}
