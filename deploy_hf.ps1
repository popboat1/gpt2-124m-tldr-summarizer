$ErrorActionPreference = "Stop"

Write-Host "Preparing Hugging Face deployment..." -ForegroundColor Cyan

# Switch to a detached HEAD so we don't mess up the main branch
git checkout --detach

# Prepend the Hugging Face YAML config to the README
$yaml = @"
---
title: GPT-2 Summarizer
emoji: 🚀
colorFrom: blue
colorTo: red
sdk: docker
pinned: false
---
"@

$content = Get-Content README.md -Raw
$newContent = "$yaml`n`n$content"
Set-Content README.md $newContent

# Commit the temporary README change
git add README.md
git commit -m "Inject Hugging Face config"

Write-Host "Pushing to Hugging Face Spaces..." -ForegroundColor Cyan
# Force push this detached state to the Hugging Face main branch
git push -f hf HEAD:main

# Go back to the real main branch and discard the detached state
git checkout main

Write-Host "Deployment successful! Your local repo is untouched." -ForegroundColor Green
