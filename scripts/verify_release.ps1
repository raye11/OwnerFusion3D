$ErrorActionPreference = "Stop"

python -m compileall -q ownerfusion3d
if ($LASTEXITCODE -ne 0) {
    throw "Python compilation check failed."
}

python -m ownerfusion3d --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "OwnerFusion3D CLI smoke test failed."
}

python -m ownerfusion3d.cli.fuse_parts --help | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Multi-part CLI smoke test failed."
}

$forbidden = @(
    "(?i)\b[A-Z]:\\Users\\",
    "(?i)\b[A-Z]:\\work\\",
    "(?i)[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"
)

$matches = Get-ChildItem -Recurse -File |
    Where-Object { $_.FullName -notmatch "\\.git\\|__pycache__|verify_release.ps1" } |
    Select-String -Pattern $forbidden -CaseSensitive:$false

if ($matches) {
    $matches | Format-Table -AutoSize
    throw "Potential identity-bearing text found in the release directory."
}

Write-Host "OwnerFusion3D anonymous release checks passed."
