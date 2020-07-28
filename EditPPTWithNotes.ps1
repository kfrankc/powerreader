param(
    [string] $PPTPath,
    [string] $CaptionPath
)

$Powerpoint = New-Object -ComObject powerpoint.application
$ppt = $Powerpoint.presentations.open($PPTPath, 2, $True, $False)

$captions = Get-Content $CaptionPath
foreach($line in $captions) {
    $arr = $line.Split(":")
    $slideNum = [int]$arr[0].Trim()
    $notes = $arr[1].Trim()

    $currentNotes = $ppt.Slides[$slideNum].NotesPage.Shapes[2].TextFrame.TextRange.Text
    $newNotes = $currentNotes += $notes

    $ppt.slides[$slideNum].NotesPage.Shapes[2].TextFrame.TextRange.Text = $newNotes
}

Sleep -Seconds 3
$ppt.SaveAs($PPTPath)
$ppt.Close()
$Powerpoint.Quit()