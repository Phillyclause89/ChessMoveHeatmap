<#
.SYNOPSIS
    A simple PowerShell script to watch the .\pgns\trainings\ and .\SQLite3Caches\QTables\ dirs during engine training.
.DESCRIPTION
    Used for the training YouTube stream here: https://www.youtube.com/live/Fh1I9DALeEc
.PARAMETER TrainingDirectory
    Relative path to the Dir that the training files to watch are saved to. By default this is pgns\trainings from repo root.
.PARAMETER QTableDirectory
    Relative path to the Dir that the q-table .db files to watch are saved to. By default this is SQLite3Caches\QTables from repo root.
.PARAMETER MaxGames
    Governs the exit logic for the script. By default, this script will continue to watch until 1000 pgn files are detected.
.PARAMETER PollIntervalSeconds
    Sleep time between pulls of the output dirs. By default this is 2 seconds.
.INPUTS
    None - No pipeline input accepted.
.OUTPUTS
    None - Prints summary of training to host.
.NOTES
    Author:         Phillyclause89
    Creation Date:  4/15/2025
    Purpose/Change: Initial script development

.EXAMPLE
    PS C:\Users\PhillyClause89\Documents\ChessMoveHeatmap> .\chmengine\stream\scripts\TrainingWatcher.ps1 -MaxGames 5 -PollIntervalSeconds 1
#>
[CmdletBinding()]
param(
    [string]$TrainingDirectory = '.\pgns\trainings\',
    [string]$QTableDirectory = '.\SQLite3Caches\QTables\',
    [int]$MaxGames = 1000,
    [int]$PollIntervalSeconds = 2
)

function Watch-TrainingGames {
    [CmdletBinding()]
    param (
        [string]$TrainingDirectory = '.\pgns\trainings\',
        [string]$QTableDirectory = '.\SQLite3Caches\QTables\',
        [int]$MaxGames = 1000,
        [int]$PollIntervalSeconds = 2
    )

    $lastSize = 0
    $lastCount = -1
    $color = 'White'

    while ((Get-ChildItem -Path $TrainingDirectory -File).Count -le $MaxGames) {
        $trainings = Get-ChildItem -Path $TrainingDirectory -File
        if ($lastCount -ne $trainings.Count) {
            Clear-Host
            $lastCount = $trainings.Count

            $stats = @{}

            $script:lastLine = ''

            $trainings | Sort-Object -Property LastWriteTime, CreationTime | ForEach-Object {
                $game = Get-Content $_.FullName
                $event_ = $game | Where-Object { $_ -match '\[Event' }
                $date = $game | Where-Object { $_ -match '\[UTCDate' }
                $time = $game | Where-Object { $_ -match '\[UTCTime' }
                $round = $game | Where-Object { $_ -match '\[Round' }
                $result = $game | Where-Object { $_ -match '\[Result' }
                $termination = $game | Where-Object { $_ -match '\[Termination' }
                $line = $game | Where-Object { $_ -match '^1\.' }

                $whiteName = (
                    $game | Where-Object { $_ -match '^\[White ' } 
                ) -replace '^\[White\s+"(.+)"\]', '$1'
                $blackName = (
                    $game | Where-Object { $_ -match '^\[Black ' } 
                ) -replace '^\[Black\s+"(.+)"\]', '$1'
                

                foreach ($player in @($whiteName, $blackName)) {
                    if (-not $stats.ContainsKey($player)) {
                        $stats[$player] = @{
                            White = @{ Wins = 0; Losses = 0; Draws = 0 }
                            Black = @{ Wins = 0; Losses = 0; Draws = 0 }
                        }
                    }
                }
                switch ($result -replace '^\[Result\s+"(.+)"\]', '$1') {
                    '1-0' {
                        # White won
                        $stats[$whiteName].White.Wins++
                        $stats[$blackName].Black.Losses++
                        $color = 'Cyan'
                    }
                    '0-1' {
                        # Black won
                        $stats[$blackName].Black.Wins++
                        $stats[$whiteName].White.Losses++
                        $color = 'Yellow'
                    }
                    default {
                        # draw for both
                        $stats[$whiteName].White.Draws++
                        $stats[$blackName].Black.Draws++
                        $color = 'Magenta'
                    }
                }

                Write-Host "$date $time $round $event_ $result $termination" -ForegroundColor $color
                $script:lastLine = $line
            }

            Write-Host "Training Games Completed: $lastCount of $MaxGames"
            foreach ($player in $stats.Keys | Sort-Object) {
                $ww = $stats[$player].White.Wins
                $wl = $stats[$player].White.Losses
                $wd = $stats[$player].White.Draws
                $bw = $stats[$player].Black.Wins
                $bl = $stats[$player].Black.Losses
                $bd = $stats[$player].Black.Draws

                # Print e.g. "CMHMEngine2: White Wins: 10 Black Wins: 8 Draws: 2"
                Write-Host "$($player):" -ForegroundColor 'Green' -NoNewline
                Write-Host " White: Wins=$ww"  -ForegroundColor 'Cyan'   -NoNewline
                Write-Host " Losses=$wl" -ForegroundColor 'Yellow' -NoNewline
                Write-Host " Draws=$wd"  -ForegroundColor 'Magenta' -NoNewline
                Write-Host " |" -ForegroundColor 'Green' -NoNewline
                Write-Host " Black: Wins=$bw"  -ForegroundColor 'Yellow' -NoNewline
                Write-Host " Losses=$bl" -ForegroundColor 'Cyan'   -NoNewline
                Write-Host " Draws=$bd"  -ForegroundColor 'Magenta'
            }
            Write-Host 'Last Completed Game Line:'
            Write-Host "$script:lastLine" -ForegroundColor $color
        }

        # Monitor Q-table size
        $fileSize = (Get-ChildItem -Path $QTableDirectory -File | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($fileSize / 1MB, 3)

        if ($sizeMB -gt $lastSize) {
            $lastSize = $sizeMB
            Write-Host "`rQ Table Size: $sizeMB MB   " -NoNewline
        }

        Start-Sleep -Seconds $PollIntervalSeconds
    }
}

# This simulates: `if __name__ == "__main__":` from python. kinda...
if ($MyInvocation.InvocationName -ne '.') {
    Watch-TrainingGames -TrainingDirectory $TrainingDirectory -QTableDirectory $QTableDirectory -MaxGames $MaxGames -PollIntervalSeconds $PollIntervalSeconds
}
# P.s. I think PowerShell is a terrible scripting language and the guy who created it is certainly not Dutch.


