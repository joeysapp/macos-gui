$script = "$env:USERPROFILE\bin\clipboard-network.ps1"

New-Item -ItemType Directory -Force "$env:USERPROFILE\bin" | Out-Null

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -STA -WindowStyle Hidden -File `"$script`" server"

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME

Register-ScheduledTask `
    -TaskName "clipboard-network-bridge" `
    -Action $action `
    -Trigger $trigger `
    -Description "Windows clipboard bridge for clipboard-network SSH push/pull handling standard input as expected" `
    -Force