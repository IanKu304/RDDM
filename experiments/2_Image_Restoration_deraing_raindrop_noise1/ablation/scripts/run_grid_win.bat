@echo off
setlocal enabledelayedexpansion

REM 使用方式：
REM   cd experiments\2_Image_Restoration_deraing_raindrop_noise1
REM   ablation\scripts\run_grid_win.bat Ian

set MEMBER=%1
if "%MEMBER%"=="" set MEMBER=member

set STEPS=120000
set T=10
set SEED=10
set DEVICE=0
set RESULTS_ROOT=results

if not exist ablation\summaries mkdir ablation\summaries
set OUTCSV=ablation\summaries\%MEMBER%.csv

for %%E in (0.01 0.02 0.04) do (
  for %%S in (0.5 1.0 2.0) do (
    set RUNNAME=GT-RAIN__img256__bs1__acc2__steps%STEPS%__schedlinear__betaEnd%%E__betaScale%%S__seed%SEED%
    echo === RUN !RUNNAME! ===

    python train_ablation.py ^
      --train_num_steps %STEPS% ^
      --sampling_timesteps %T% ^
      --seed %SEED% ^
      --device %DEVICE% ^
      --amp ^
      --beta_schedule linear ^
      --beta_end %%E ^
      --beta_scale %%S ^
      --run_name !RUNNAME! ^
      --results_root %RESULTS_ROOT%

    REM 可選：每次 train 完直接 test+算 metrics
    REM python test_ablation.py --run_dir %RESULTS_ROOT%\!RUNNAME! --ckpt latest --sampling_timesteps %T% --eval_csv %OUTCSV% --compute_fid --compute_lpips
  )
)

endlocal
