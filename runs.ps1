$inputDirectory = "./"
$outputFile = "combined.py"

# Get all Python scripts in the directory
$pythonFiles = Get-ChildItem -Path $inputDirectory -Filter *.py

# Initialize the content for the merged script
$mergedScriptContent = ""

foreach ($file in $pythonFiles) {
    # Read the content of the current Python script
    $fileContent = Get-Content -Path $file.FullName

    # Add a comment to indicate the source file in the merged script
    $mergedScriptContent += "`n# Start of $($file.Name)`n"
    $mergedScriptContent += $fileContent
    $mergedScriptContent += "`n# End of $($file.Name)`n"
}

# Write the merged content to the output file
Set-Content -Path $outputFile -Value $mergedScriptContent

Write-Host "Merged script created at $outputFile"