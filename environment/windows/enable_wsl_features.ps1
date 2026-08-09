#Requires -RunAsAdministrator

$ErrorActionPreference = 'Stop'

$features = @(
    'Microsoft-Windows-Subsystem-Linux',
    'VirtualMachinePlatform'
)

foreach ($feature in $features) {
    Write-Host "Enabling Windows feature: $feature"
    & "$env:SystemRoot\System32\dism.exe" /online /enable-feature "/featurename:$feature" /all /norestart

    if ($LASTEXITCODE -notin @(0, 3010)) {
        throw "DISM failed for $feature with exit code $LASTEXITCODE"
    }
}

Write-Host ''
Write-Host 'WSL prerequisites are enabled. Restart Windows before installing Ubuntu 22.04.'
