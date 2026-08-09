param(
    [ValidateSet('start', 'status', 'stop')]
    [string]$Action = 'start'
)

$ErrorActionPreference = 'Stop'
$distro = 'Ubuntu-22.04'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).Path

if ($projectRoot -notmatch '^([A-Za-z]):\\(.*)$') {
    throw "Unsupported Windows project path: $projectRoot"
}
$drive = $Matches[1].ToLowerInvariant()
$relativePath = $Matches[2].Replace('\', '/')
$linuxProjectRoot = "/mnt/$drive/$relativePath"

$linuxScript = "$linuxProjectRoot/environment/ubuntu/jupyter_server.sh"
& wsl -d $distro -- bash $linuxScript $Action

if ($LASTEXITCODE -ne 0) {
    throw "Jupyter action '$Action' failed with exit code $LASTEXITCODE."
}
