param(
  [string]$Member = "member",
  [int]$Steps = 120000,
  [int]$SamplingTimesteps = 10,
  [int]$Seed = 10,
  [string]$Device = "0",
  [string]$ResultsRoot = "results"
)

# 使用方式（在 experiment 資料夾）：
# powershell -ExecutionPolicy Bypass -File .\ablation\scripts\run_grid_win.ps1 -Member Ian

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$expDir = Resolve-Path (Join-Path $scriptDir "..\..")
Set-Location $expDir

$betaEnds = @(0.01, 0.02, 0.04)
$betaScales = @(0.5, 1.0, 2.0)

$outCsv = Join-Path $expDir ("ablation\summaries\" + $Member + ".csv")
New-Item -ItemType Directory -Force -Path (Split-Path $outCsv) | Out-Null

foreach ($be in $betaEnds) {
  foreach ($bs in $betaScales) {
    $runName = "GT-RAIN__img256__bs1__acc2__steps$Steps__schedlinear__betaEnd$be__betaScale$bs__seed$Seed"
    Write-Host "=== RUN $runName ==="

    python .\train_ablation.py `
      --train_num_steps $Steps `
      --sampling_timesteps $SamplingTimesteps `
      --seed $Seed `
      --device $Device `
      --amp `
      --beta_schedule linear `
      --beta_end $be `
      --beta_scale $bs `
      --run_name $runName `
      --results_root $ResultsRoot

    # 可選：每次 train 完直接 test+算 metrics
    # python .\test_ablation.py --run_dir (Join-Path $ResultsRoot $runName) --ckpt latest --sampling_timesteps $SamplingTimesteps --eval_csv $outCsv --compute_fid --compute_lpips
  }
}
