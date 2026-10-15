param([string]$RunFolder = 'checkpoints/reduced_mpc_20_20261012')
$repoPath = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$runPath = Join-Path $repoPath $RunFolder
$Host.UI.RawUI.WindowTitle = 'MPC Comparison - Live Progress'
while ($true) {
    try {
        $result = Get-Content -LiteralPath (Join-Path $runPath 'results.json') -Raw | ConvertFrom-Json
        $lastLine = Get-Content -LiteralPath (Join-Path $runPath 'planning_per_timestep.csv') -Tail 1
        $last = $lastLine | ConvertFrom-Csv -Header method,episode,timestep,horizon,planning_seconds,solver_success,success,state_dist
        $finished = @($result.planning).Count
        $total = @($result.models.PSObject.Properties).Count * $result.settings.episodes
        $rows = foreach ($property in $result.models.PSObject.Properties) {
            $episodes = @($result.planning | Where-Object { $_.method -eq $property.Name })
            $successes = @($episodes | Where-Object { $_.final_success }).Count
            [PSCustomObject]@{
                Method = $property.Name
                Completed = "$($episodes.Count)/$($result.settings.episodes)"
                Successes = $successes
                'Success rate' = if ($episodes.Count) { '{0:P1}' -f ($successes / $episodes.Count) } else { '-' }
            }
        }
        Clear-Host
        Write-Host ('MPC comparison | Updated: ' + (Get-Date -Format 'HH:mm:ss')) -ForegroundColor Cyan
        Write-Host ('Status: {0} | Episodes: {1}/{2} | Progress: {3:P1}' -f $result.status,$finished,$total,($finished/[double]$total))
        Write-Host ('Current: {0} | Episode: {1} | Timestep: {2}/200 | Horizon: {3}' -f $last.method,([int]$last.episode+1),([int]$last.timestep+1),$last.horizon)
        Write-Host ('Latest planning: {0:N1} ms | Goal distance: {1:N2} pixels' -f ([double]$last.planning_seconds*1000),[double]$last.state_dist)
        $rows | Format-Table -AutoSize | Out-Host
        Write-Host 'Results are partial until Status is complete. Refresh: 5 seconds. Close this window to stop monitoring.'
        if ($result.status -eq 'complete') { break }
    } catch {
        Write-Host ('Waiting for a readable result snapshot: ' + $_.Exception.Message)
    }
    Start-Sleep -Seconds 5
}
