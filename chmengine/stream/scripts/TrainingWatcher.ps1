<#
.SYNOPSIS
    A simple PowerShell script to watch the .\pgns\trainings\ and .\SQLite3Caches\QTables\ dirs during engine training.
.DESCRIPTION
    Used for the training YouTube stream here: https://www.youtube.com/watch?v=_-JySFYZhjU
.PARAMETER TrainingDirectory
    Relative path to the Dir that the training files to watch are saved to. By default this is pgns\trainings from repo root.
.PARAMETER QTableDirectory
    Relative path to the Dir that the q-table .db files to watch are saved to. By default this is SQLite3Caches\QTables from repo root.
.PARAMETER PollIntervalSeconds
    Sleep time between pulls of the output dirs. By default this is 2 seconds.
.PARAMETER InitialElo
    A hashtable of player names to their initial Elo ratings. This is used to calculate the predicted Elo of each player.
    The hashtable should be in the format:
        $InitialElo['Alice'] = @{ White=1600; Black=1580 }
    If a player is not found in the hashtable, the PoolAvg value will be used instead.
.PARAMETER PoolAvg
    The average Elo of the pool of players. This is used to calculate the predicted Elo of each player. By default this is 1500.
.PARAMETER MaxGames
    Governs the exit logic for the script. By default, this is set to 0 meaing the script will run indefinitely.
    If this is set to a positive integer, the script will exit after that many games have been processed.
.PARAMETER Infinite
    If this switch is set, the script will run indefinitely, even if the MaxGames limit is reached. This is useful for long-running training sessions.
    By default, this is set to $true as MaxGames default value is 0.

.INPUTS
    None - No pipeline input accepted.
.OUTPUTS
    None - Prints summary of training to host.
.NOTES
    Author:         Phillyclause89
    Creation Date:  4/15/2025

.EXAMPLE
    ```PowerShell
    PS C:\Users\PhillyClause89\Documents\ChessMoveHeatmap> .\chmengine\stream\scripts\TrainingWatcher.ps1 -MaxGames 5 -PollIntervalSeconds 1 -PoolAvg 1000 -InitialElo @{'PhillyClause89' = @{ White=1700; Black=1680 }}
    ```
        `-MaxGames 5`                                                   : This will watch the training directory for 5 games, 
        `-PollIntervalSeconds 1`                                        : polling every second, 
        `-PoolAvg 1000`                                                 : and using an average Elo of 1000 for all players not defined in the -InitialElo mapping. 
        `-InitialElo @{'PhillyClause89' = @{ White=1700; Black=1680 }}` : The initial Elo for PhillyClause89 will be set to 1700 for White and 1680 for Black.
#>
[CmdletBinding()]
param(
    [Parameter(
        Mandatory = $false,
        Position = 0
    )]
    [string]$TrainingDirectory = '.\pgns\trainings\',
    [Parameter(
        Mandatory = $false,
        Position = 1
    )]
    [string]$QTableDirectory = '.\SQLite3Caches\QTables\',
    [Parameter(
        Mandatory = $false,
        Position = 2
    )]
    [int]$PollIntervalSeconds = 2,
    [Parameter(
        Mandatory = $false,
        Position = 3
    )]    
    [hashtable]$InitialElo = @{
        'CMHMEngine'     = @{ White = 100; Black = 50 }
        'CMHMEngine2'    = @{ White = 350; Black = 300 }
        'PhillyClause89' = @{ White = 1600; Black = 1550 }
    },
    [Parameter(
        Mandatory = $false,
        Position = 4
    )]
    [double]$PoolAvg = (
        & {
            if ($InitialElo.Count -eq 0) { 1500.0 }
            else { [math]::Round(($InitialElo.Values | Measure-Object -Property White -Sum).Sum / $InitialElo.Count, 0) }
        }
    ),
    [Parameter(
        Mandatory = $false,
        Position = 5
    )]
    [int]$MaxGames = 0,
    [Parameter(
        Mandatory = $false
    )]
    [switch]$Infinite = (& { if ($MaxGames -lt 1) { $true } else { $false } })
)
# Set default values for script variables
[bool]$script:_init_ = $true
[double]$script:lastSize = 0.0
[string]$script:lastLine = ''
[string]$script:TrainingDirectory = $TrainingDirectory
[string]$script:QTableDirectory = $QTableDirectory
[int]$script:MaxGames = $MaxGames
[int]$script:PollIntervalSeconds = $PollIntervalSeconds
[hashtable]$script:InitialElo = $InitialElo
[double]$script:PoolAvg = $PoolAvg
[bool]$script:Infinite = $Infinite

function Get-PredictedEloPerSide {
    <#
    .SYNOPSIS
        Estimate White- and Black-side Elo for each player, with optional per-player initial Elos.
    .DESCRIPTION
        This function estimates the Elo rating for each player based on their win/loss/draw statistics.
        It can also take an initial Elo rating for each player, which will be used as a base for the calculations.
    .PARAMETER Stats
        Hashtable of player → @{ White = @{Wins,Losses,Draws}; Black = @{...} }.
    .PARAMETER PoolAvg
        Fallback “field average” Elo if no InitialElo is provided for a player/side. Default 1500.
    .PARAMETER InitialElo
        OPTIONAL hashtable of player → @{ White=[int]; Black=[int] }.  Missing entries fall back to PoolAvg.
    .OUTPUTS
        Hashtable of player → @{ White=[double]; Black=[double] }.
    #>
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)]
        [hashtable]$Stats,
        [int]$PoolAvg = $script:PoolAvg,
        [hashtable]$InitialElo = $script:InitialElo
    )

    $elos = @{}

    foreach ($player in $Stats.Keys) {
        # Unpack stats
        $wW = $Stats[$player].White.Wins
        $lW = $Stats[$player].White.Losses
        $dW = $Stats[$player].White.Draws
        $wB = $Stats[$player].Black.Wins
        $lB = $Stats[$player].Black.Losses
        $dB = $Stats[$player].Black.Draws

        # Determine base Elos
        if ($InitialElo.ContainsKey($player) -and $InitialElo[$player].ContainsKey('White')) {
            $baseWhite = $InitialElo[$player].White
        }
        else {
            $baseWhite = $PoolAvg
        }
        if ($InitialElo.ContainsKey($player) -and $InitialElo[$player].ContainsKey('Black')) {
            $baseBlack = $InitialElo[$player].Black
        }
        else {
            $baseBlack = $PoolAvg
        }

        # Helper to compute side Elo given base
        function Get-SideElo {
            param(
                [int]$wins,
                [int]$losses,
                [int]$draws,
                [double]$baseElo
            )
            $games = $wins + $losses + $draws
            if ($games -eq 0) { return $null }

            $score = $wins + 0.5 * $draws
            $S = $score / $games

            if ($S -ge 1.0) {
                $delta = 800
            }
            elseif ($S -le 0.0) {
                $delta = -800
            }
            else {
                $delta = -400 * [math]::Log10((1 / $S) - 1)
            }
            return [math]::Round($baseElo + $delta, 1)
        }

        $eloWhite = Get-SideElo -wins $wW -losses $lW -draws $dW -baseElo $baseWhite
        $eloBlack = Get-SideElo -wins $wB -losses $lB -draws $dB -baseElo $baseBlack

        $elos[$player] = @{
            White = $eloWhite
            Black = $eloBlack
        }
    }

    return $elos
}

function Get-QTableColor {
    [CmdletBinding()]
    param (
        [double]$sizeMB = (Get-QTableSize -QTableDirectory $script:QTableDirectory),
        [string]$QTableDirectory = $script:QTableDirectory
    )

    # Resolve drive letter or root path
    $driveRoot = (Get-Item -Path $QTableDirectory).PSDrive.Root
    $driveInfo = Get-PSDrive | Where-Object { $_.Root -eq $driveRoot }

    if (-not $driveInfo) {
        return 'Gray'  # Fallback in case of error
    }

    $freeSpaceMB = [math]::Round($driveInfo.Free / 1MB, 3)

    if ($freeSpaceMB -lt 500) {
        return 'DarkRed'  # Danger zone
    }

    # Ratio of Q-table size to remaining free space
    $usageRatio = $sizeMB / $freeSpaceMB

    # Gradient logic (adjust thresholds as needed)
    if ($usageRatio -lt 0.1) {
        return 'Green'
    }
    elseif ($usageRatio -lt 0.3) {
        return 'Yellow'
    }
    elseif ($usageRatio -lt 0.5) {
        return 'DarkYellow'
    }
    elseif ($usageRatio -lt 0.75) {
        return 'Magenta'
    }
    else {
        return 'Red'
    }
}

function Watch-TrainingGames {
    [CmdletBinding()]
    param (
        [string]$TrainingDirectory = $script:TrainingDirectory,
        [string]$QTableDirectory = $script:QTableDirectory,
        [int]$MaxGames = $script:MaxGames,
        [bool]$Infinite = $script:Infinite,
        [int]$PollIntervalSeconds = $script:PollIntervalSeconds,
        [int]$PoolAvg = $script:PoolAvg,
        [hashtable]$InitialElo = $script:InitialElo
    )
    $script:_init_ = $true
    $script:lastSize = Get-QTableSize -QTableDirectory $QTableDirectory
    $lastCount = -1
    $color = 'White'

    while (((Get-ChildItem -Path $TrainingDirectory -File).Count -le $MaxGames) -or ($Infinite) -or ($script:_init_)) {
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

            $elos = Get-PredictedEloPerSide -Stats $stats -PoolAvg $PoolAvg -InitialElo $InitialElo

            Write-Host 'Training Games Completed: ' -ForegroundColor 'DarkGreen' -NoNewline
            if ($Infinite) {
                Write-Host "$lastCount" -ForegroundColor 'Green'
            }
            else {
                Write-Host "$lastCount of $MaxGames" -ForegroundColor 'Green'
            }
            
            

            foreach ($player in $stats.Keys | Sort-Object) {
                $ww = $stats[$player].White.Wins
                $wl = $stats[$player].White.Losses
                $wd = $stats[$player].White.Draws
                $we = $elos[$player].White
                $bw = $stats[$player].Black.Wins
                $bl = $stats[$player].Black.Losses
                $bd = $stats[$player].Black.Draws
                $be = $elos[$player].Black

                # Print e.g. "CMHMEngine2: White Wins: 10 Black Wins: 8 Draws: 2"
                Write-Host "$($player)" -ForegroundColor 'DarkGreen' -NoNewline
                Write-Host " (White): Wins=$ww"  -ForegroundColor 'Cyan'   -NoNewline
                Write-Host ', ' -ForegroundColor 'Cyan' -NoNewline
                Write-Host "Losses=$wl" -ForegroundColor 'Yellow' -NoNewline
                Write-Host ', ' -ForegroundColor 'Cyan' -NoNewline
                Write-Host "Draws=$wd"  -ForegroundColor 'Magenta' -NoNewline
                Write-Host ', ' -ForegroundColor 'Cyan' -NoNewline
                Write-Host "Predicted Elo=$we" -ForegroundColor 'Cyan'
                Write-Host "$($player)" -ForegroundColor 'DarkGreen' -NoNewline
                Write-Host " (Black): Wins=$bw"  -ForegroundColor 'Yellow' -NoNewline
                Write-Host ', ' -ForegroundColor 'Yellow' -NoNewline
                Write-Host "Losses=$bl" -ForegroundColor 'Cyan'   -NoNewline
                Write-Host ', ' -ForegroundColor 'Yellow' -NoNewline
                Write-Host "Draws=$bd"  -ForegroundColor 'Magenta' -NoNewline
                Write-Host ', ' -ForegroundColor 'Yellow' -NoNewline
                Write-Host "Predicted Elo=$be" -ForegroundColor 'Yellow' 
            }
            Write-Host 'Last Completed Game Line: ' -ForegroundColor 'DarkGreen' -NoNewline
            Write-Host "$script:lastLine" -ForegroundColor $color
        }

        # Monitor Q-table size
        $script:lastSize = Write-QTableSize -sizeMB (
            Get-QTableSize -QTableDirectory $QTableDirectory
        ) -lastSize $script:lastSize -PollIntervalSeconds $PollIntervalSeconds

        Start-Sleep -Seconds $PollIntervalSeconds
    }
    Write-Host
}

function Write-QTableSize {
    [CmdletBinding()]
    param (
        [double]$sizeMB = (Get-QTableSize -QTableDirectory $script:QTableDirectory),
        [double]$lastSize = $script:lastSize,
        [string]$QTableDirectory = $script:QTableDirectory,
        [int]$PollIntervalSeconds = $script:PollIntervalSeconds
    )
    if (($sizeMB -ne $lastSize) -or ($script:_init_)) {
        $script:_init_ = $false
        $growthRate = ($sizeMB - $lastSize) / $PollIntervalSeconds
        $growthRateMB = [math]::Round($growthRate, 3)
        $lastSize = $sizeMB
        Write-Host "`rQ Table Size: $sizeMB MB ($($growthRateMB.ToString('+#.000;-#.000;0.000')) MB/s)   " -ForegroundColor (
            Get-QTableColor -sizeMB $sizeMB -QTableDirectory $QTableDirectory
        ) -NoNewline
    }
    return $lastSize
}

function Get-QTableSize {
    [CmdletBinding()]
    param (
        [string]$QTableDirectory = $script:QTableDirectory,
        [int]$SizeSpec = 1MB,
        [int]$SizeDecimal = 3
    )
    $fileSize = (Get-ChildItem -Path $QTableDirectory -File | Measure-Object -Property Length -Sum).Sum
    return [math]::Round($fileSize / $SizeSpec, $SizeDecimal)
}

# This simulates: `if __name__ == "__main__":` from python. kinda...
if ($MyInvocation.InvocationName -ne '.') {
    $WTGArgs = @{
        TrainingDirectory   = $script:TrainingDirectory
        QTableDirectory     = $script:QTableDirectory
        MaxGames            = $script:MaxGames
        PollIntervalSeconds = $script:PollIntervalSeconds
        InitialElo          = $script:InitialElo
        PoolAvg             = $script:PoolAvg
        Infinite            = $script:Infinite
    }
    Watch-TrainingGames @WTGArgs
}
# P.s. I think PowerShell is a terrible scripting language and the guy who created it is certainly not Dutch.


