# run_interp_ablation.ps1 - interp-split ablation for aug/ms generalization
# train on even-phi angles, test on odd-phi angles (234/234)
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

# 2x2 ablation on interp split, w64/150ep
Run "interp_plain"  @("exp_improve.py", "--split", "interp", "--width", "64", "--epochs", "150",
                     "--nffft", "--n-angles", "40", "--tag", "interp_plain")
Run "interp_aug"    @("exp_improve.py", "--split", "interp", "--aug", "--width", "64", "--epochs", "150",
                     "--tag", "interp_aug")
Run "interp_ms"     @("exp_improve.py", "--split", "interp", "--ms", "0.3", "--width", "64", "--epochs", "150",
                     "--tag", "interp_ms")
Run "interp_augms"  @("exp_improve.py", "--split", "interp", "--aug", "--ms", "0.3", "--width", "64", "--epochs", "150",
                     "--nffft", "--n-angles", "40", "--tag", "interp_augms")

Write-Output "===== ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====="
