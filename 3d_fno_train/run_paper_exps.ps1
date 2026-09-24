# run_paper_exps.ps1 - sequential paper experiments (suggestions 1/3/4)
# Launched via Start-Process hidden; logs redirected to results/exp_run_all.log
$ErrorActionPreference = "Continue"
$py = "F:/miniconda3/envs/isaac311/python.exe"
Set-Location $PSScriptRoot

function Run($name, $a) {
    Write-Output "===== START $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
    & $py @a 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Output "!!!!! FAILED $name exit=$LASTEXITCODE"
    } else {
        Write-Output "===== DONE $name $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
    }
}

# P0 suggestion 1: baseline comparison (F-FNO vs MLP vs U-Net vs DeepONet)
Run "exp_baselines" @("exp_baselines.py", "--model", "all", "--split", "full")

# P0 suggestion 3: augmentation / multi-scale loss ablation (w64/150ep)
Run "exp_improve_aug"   @("exp_improve.py", "--aug", "--width", "64", "--epochs", "150", "--tag", "aug_w64")
Run "exp_improve_ms"    @("exp_improve.py", "--ms", "0.3", "--width", "64", "--epochs", "150", "--tag", "ms_w64")
Run "exp_improve_augms" @("exp_improve.py", "--aug", "--ms", "0.3", "--width", "64", "--epochs", "150", "--tag", "augms_w64")

# P0 suggestion 3: improved model + ensemble + NFFFT far-field error propagation
Run "exp_improve_nffft" @("exp_improve.py", "--aug", "--ms", "0.3", "--seeds", "3",
                          "--width", "64", "--epochs", "150", "--nffft",
                          "--n-angles", "40", "--tag", "ens3_nffft")

# P1 suggestion 4: size-frequency duality generalization
Run "exp_extension" @("exp_extension.py", "--step", "8", "--width", "64", "--epochs", "150")

Write-Output "===== ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
