# PowerShell script to copy today's folders from source to destination
param(
    [string]$SourceRoot = "C:\DataStore\TradingSystems",
    [string]$DestinationRoot = "H:\DataStore\TradingSystems"
)

# Get today's date for comparison
$Today = Get-Date -Format "yyyy-MM-dd"

Write-Host "Starting folder copy process for date: $Today" -ForegroundColor Green

# Get all signal_name directories from the source
$SourcePath = "$SourceRoot"
$SignalDirectories = Get-ChildItem -Path $SourcePath -Directory | Where-Object { 
    Test-Path (Join-Path $_.FullName "prodMHS\Signals\history") 
}

if ($SignalDirectories.Count -eq 0) {
    Write-Host "No signal directories found with the expected structure." -ForegroundColor Red
    exit 1
}

Write-Host "Found $($SignalDirectories.Count) signal directories" -ForegroundColor Yellow

foreach ($SignalDir in $SignalDirectories) {
    $SignalName = $SignalDir.Name
    Write-Host "`nProcessing signal: $SignalName" -ForegroundColor Cyan
    
    # Define source and destination paths
    $SourceHistoryPath = Join-Path $SignalDir.FullName "prodMHS\Signals\history"
    $DestinationHistoryPath = Join-Path $DestinationRoot "$SignalName\prodMHS\Signals\history"
    
    # Check if source history directory exists
    if (-not (Test-Path $SourceHistoryPath)) {
        Write-Host "  Source history path not found: $SourceHistoryPath" -ForegroundColor Red
        continue
    }
    
    # Get folders created today in the history directory
    $TodayFolders = Get-ChildItem -Path $SourceHistoryPath -Directory | Where-Object {
        $_.CreationTime.Date -eq (Get-Date).Date
    }
    
    if ($TodayFolders.Count -eq 0) {
        Write-Host "  No folders created today found in: $SourceHistoryPath" -ForegroundColor Yellow
        continue
    }
    
    Write-Host "  Found $($TodayFolders.Count) folder(s) created today" -ForegroundColor Green
    
    # Create destination directory structure if it doesn't exist
    if (-not (Test-Path $DestinationHistoryPath)) {
        Write-Host "  Creating destination directory: $DestinationHistoryPath" -ForegroundColor Blue
        try {
            New-Item -ItemType Directory -Path $DestinationHistoryPath -Force | Out-Null
            Write-Host "  Successfully created destination directory" -ForegroundColor Green
        }
        catch {
            Write-Host "  Failed to create destination directory: $($_.Exception.Message)" -ForegroundColor Red
            continue
        }
    }
    
    # Copy each folder created today
    foreach ($Folder in $TodayFolders) {
        $SourceFolderPath = $Folder.FullName
        $DestinationFolderPath = Join-Path $DestinationHistoryPath $Folder.Name
        
        Write-Host "  Copying: $($Folder.Name)" -ForegroundColor White
        
        try {
            # Copy the folder and all its contents
            Copy-Item -Path $SourceFolderPath -Destination $DestinationFolderPath -Recurse -Force
            Write-Host "    ✓ Successfully copied to: $DestinationFolderPath" -ForegroundColor Green
        }
        catch {
            Write-Host "    ✗ Failed to copy: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

Write-Host "`nFolder copy process completed!" -ForegroundColor Green

# Optional: Display summary
Write-Host "`n=== SUMMARY ===" -ForegroundColor Magenta
Write-Host "Date processed: $Today" -ForegroundColor White
Write-Host "Total signal directories processed: $($SignalDirectories.Count)" -ForegroundColor White
Write-Host "Source root: $SourceRoot" -ForegroundColor White
Write-Host "Destination root: $DestinationRoot" -ForegroundColor White
