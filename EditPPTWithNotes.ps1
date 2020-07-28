param(
    [string] $PPTPath,
    [string] $CaptionPath
)

$Powerpoint = New-Object -ComObject powerpoint.application
$ppt = $Powerpoint.presentations.open($PPTPath, 2, $True, $False)
foreach($slide in $ppt.slides){
    $slide.NotesPage.Shapes[2].TextFrame.TextRange.Text = $slide.NotesPage.Shapes[2].TextFrame.TextRange.Text += "hello"
}
Sleep -Seconds 3
$ppt.SaveAs($PPTPath)
$ppt.Close()
$Powerpoint.Quit()