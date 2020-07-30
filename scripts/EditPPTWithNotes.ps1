param(
    [string] $PPTPath="input/slides.pptx",
    [string] $CaptionPath="output/mappedcaptions.txt"
)

$cwd = Get-Location
$PPTPath = [String]$cwd + "/" + $PPTPath
$CaptionPath = [String]$cwd + "/" + $CaptionPath
$Powerpoint = New-Object -ComObject powerpoint.application
$ppt = $Powerpoint.presentations.open($PPTPath, 2, $True, $False)

$captions = Get-Content $CaptionPath
foreach($line in $captions) {
    $arr = $line.Split(":")
    $slideNum = [int]$arr[0].Trim()
    $notes = $arr[1].Trim()

    $currentNotes = $ppt.Slides.Item($slideNum).NotesPage.Shapes.Item(2).TextFrame.TextRange.Text
    $newNotes = $currentNotes += $notes

    $ppt.Slides.Item($slideNum).NotesPage.Shapes.Item(2).TextFrame.TextRange.Text = $newNotes
}

Sleep -Seconds 3
$ppt.SaveAs("../output/SlidesWithCaptions.pptx")
$ppt.Close()
$Powerpoint.Quit()