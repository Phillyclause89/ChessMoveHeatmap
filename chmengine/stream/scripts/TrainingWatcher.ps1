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
    [string]$TrainingDirectory = ".\pgns\trainings\",
    [string]$QTableDirectory = ".\SQLite3Caches\QTables\",
    [int]$MaxGames = 1000,
    [int]$PollIntervalSeconds = 2
)

function Watch-TrainingGames {
    [CmdletBinding()]
    param (
        [string]$TrainingDirectory = ".\pgns\trainings\",
        [string]$QTableDirectory = ".\SQLite3Caches\QTables\",
        [int]$MaxGames = 1000,
        [int]$PollIntervalSeconds = 2
    )

    $lastSize = 0
    $lastCount = -1

    while ((Get-ChildItem -Path $TrainingDirectory -File).Count -le $MaxGames) {
        $trainings = Get-ChildItem -Path $TrainingDirectory -File
        if ($lastCount -ne $trainings.Count) {
            Clear-Host
            $lastCount = $trainings.Count

            $termTally = @{
                "1-0" = 0
                "0-1" = 0
                "*"   = 0
            }

            $lastLine = ""

            $trainings | Sort-Object -Property CreationTime | ForEach-Object {
                $game = Get-Content $_.FullName
                $date = $game | Where-Object { $_ -match '\[UTCDate' }
                $time = $game | Where-Object { $_ -match '\[UTCTime' }
                $round = $game | Where-Object { $_ -match '\[Round' }
                $result = $game | Where-Object { $_ -match '\[Result' }
                $termination = $game | Where-Object { $_ -match '\[Termination' }
                $line = $game | Where-Object { $_ -match '^1\.' }

                if ($result -match "0-1") {
                    $color = "Yellow"
                    $termTally['0-1']++
                }
                elseif ($result -match "1-0") {
                    $color = "Cyan"
                    $termTally['1-0']++
                }
                else {
                    $color = "Magenta"
                    $termTally['*']++
                }

                Write-Host "$date $time $round $result $termination" -ForegroundColor $color
                $lastLine = $line
            }

            Write-Host "Training Games Completed: $lastCount of $MaxGames"
            Write-Host "White Wins: $( $termTally.'1-0' ) " -ForegroundColor 'Cyan' -NoNewline
            Write-Host "Black Wins: $( $termTally.'0-1' ) " -ForegroundColor 'Yellow' -NoNewline
            Write-Host "Draws: $( $termTally.'*' )" -ForegroundColor 'Magenta'
            Write-Host "Last Completed Game Line:"
            Write-Host "$lastLine" -ForegroundColor $color
        }

        # Monitor Q-table size
        $fileSize = (Get-ChildItem -Path $QTableDirectory -File | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($fileSize / 1MB, 3)

        if ($sizeMB -gt $lastSize) {
            $lastSize = $sizeMB
            Write-Host "`rQ Table Size: $sizeMB MB " -NoNewline
        }

        Start-Sleep -Seconds $PollIntervalSeconds
    }
}

# This simulates: `if __name__ == "__main__":` from python. kinda...
if ($MyInvocation.InvocationName -ne '.') {
    Watch-TrainingGames -TrainingDirectory $TrainingDirectory -QTableDirectory $QTableDirectory -MaxGames $MaxGames -PollIntervalSeconds $PollIntervalSeconds
}
# P.s. I think PowerShell is a terrible scripting language and the guy who created it is certainly not Dutch.


