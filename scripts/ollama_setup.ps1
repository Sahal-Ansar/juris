# Set up the local model Juris uses when no paid API key is configured (D-007, D-041).
#
#   powershell -ExecutionPolicy Bypass -File scripts/ollama_setup.ps1
#
# Needs Ollama for Windows (winget install Ollama.Ollama). Pulls qwen2.5:7b (~4.7 GB) and
# creates juris-qwen2.5-7b, the same weights with a 16k-token context: Ollama's OpenAI-
# compatible endpoint cannot set the context per request, and the default is too short for
# the evidence-heavy prompts of B1/B2. Then add the four Ollama lines from .env.example to .env.

$ErrorActionPreference = "Stop"
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) { $ollama = "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" }
if (-not (Test-Path $ollama)) { throw "Ollama is not installed (winget install Ollama.Ollama)" }

try { Invoke-RestMethod http://localhost:11434/api/version -TimeoutSec 3 | Out-Null }
catch { Start-Process $ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep -Seconds 5 }

& $ollama pull qwen2.5:7b
$modelfile = Join-Path $env:TEMP "juris-qwen2.5-7b.Modelfile"
"FROM qwen2.5:7b`nPARAMETER num_ctx 16384`n" | Set-Content -Path $modelfile -Encoding ascii
& $ollama create juris-qwen2.5-7b -f $modelfile
& $ollama list
