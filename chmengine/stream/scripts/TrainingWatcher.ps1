<#
.SYNOPSIS
    <Overview of script>
.DESCRIPTION
    <Brief description of script>
.PARAMETER TrainingDirectory
    <Brief description of parameter input required. Repeat this attribute if required>
.PARAMETER QTablePath
    <Brief description of parameter input required. Repeat this attribute if required>
.PARAMETER MaxGames
    <Brief description of parameter input required. Repeat this attribute if required>
.PARAMETER PollIntervalSeconds
    <Brief description of parameter input required. Repeat this attribute if required>
.INPUTS
    <Inputs if any, otherwise state None>
.OUTPUTS
    <Outputs if any, otherwise state None - example: Log file stored in C:\Windows\Temp\<name>.log>
.NOTES
    Version:        1.0
    Author:         Phillyclause89
    Creation Date:  4/15/2025
    Purpose/Change: Initial script development

.EXAMPLE
    <Example goes here. Repeat this attribute for more than one example>
#>
[CmdletBinding()]
param(
    [string]$TrainingDirectory = ".\pgns\trainings\",
    [string]$QTablePath = ".\SQLite3Caches\QTables\",
    [int]$MaxGames = 1000,
    [int]$PollIntervalSeconds = 2
)

function Watch-TrainingGames {
    [CmdletBinding()]
    param (
        [string]$TrainingDirectory = ".\pgns\trainings\",
        [string]$QTablePath = ".\SQLite3Caches\QTables\",
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
        $fileSize = (Get-ChildItem -Path $QTablePath -File | Measure-Object -Property Length -Sum).Sum
        $sizeMB = [math]::Round($fileSize / 1MB, 3)

        if ($sizeMB -gt $lastSize) {
            $lastSize = $sizeMB
            Write-Host "`rQ Table Size: $sizeMB MB " -NoNewline
        }

        Start-Sleep -Seconds $PollIntervalSeconds
    }
}

Watch-TrainingGames -TrainingDirectory $TrainingDirectory -QTablePath $QTablePath -MaxGames $MaxGames -PollIntervalSeconds $PollIntervalSeconds
