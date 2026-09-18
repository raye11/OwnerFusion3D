$ErrorActionPreference = "Stop"

python -m compileall -q ownerfusion3d
python -m ownerfusion3d --help | Out-Null
python -m ownerfusion3d.cli.fuse_parts --help | Out-Null

$forbidden = @(
    "Mengzheng",
    "Hengzhou",
    "Qiu Lu",
    "github.com"
)

$matches = Get-ChildItem -Recurse -File |
    Where-Object { $_.FullName -notmatch "\\.git\\|__pycache__|verify_release.ps1" } |
    Select-String -Pattern $forbidden -SimpleMatch -CaseSensitive:$false

if ($matches) {
    $matches | Format-Table -AutoSize
    throw "Potential identity-bearing text found in the release directory."
}

Write-Host "OwnerFusion3D anonymous release checks passed."
