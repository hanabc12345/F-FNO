-- =============================================================================
-- test_connect.lua
-- FEKO 2026 Lua 连通性测试：导入清理后的 STL → 频率 → 平面波 → 远/近场 → 求解
-- 目的：验证清理后的 F-16 网格可通过 MoM 求解
-- =============================================================================

application = cf.Application.GetInstance()
project = application:NewProject()

-- 1. 导入清理后的 STL 网格
local stl_path = [[f:\MyWorkSpace\UAVGame\3d_feko_run\f16_refined.stl]]
local ok, err = pcall(function()
    local meshes = project.Importer.MeshImporter:Import(stl_path)
    print("[OK] Import STL, count =", #meshes)
end)
if not ok then print("[FAIL] Import STL:", err) end

-- 2. 设置频率 3 GHz
local freq_ok = false
local freq_err = ""
for _, fn in ipairs({
    function() project.Contents.SolutionConfigurations.GlobalFrequency.Start = "3e9" end,
    function() project.Contents.SolutionConfigurations[1].Frequency.Start = "3e9" end,
}) do
    ok, err = pcall(fn)
    if ok then freq_ok = true; print("[OK] Frequency set"); break else freq_err = err end
end
if not freq_ok then print("[FAIL] Frequency:", freq_err) end

-- 3. 添加平面波（theta=90°, phi=0°）
ok, err = pcall(function()
    local pw = project.Contents.SolutionConfigurations.GlobalSources:AddPlaneWave(90, 0)
    print("[OK] PlaneWave added:", pw.Label)
end)
if not ok then print("[FAIL] PlaneWave:", err) end

-- 4. 添加远场请求（theta 0..180, phi 0..360, 间隔 30°）
ok, err = pcall(function()
    local config = project.Contents.SolutionConfigurations[1]
    local ff = config.FarFields:Add(0, 0, 180, 360, 30, 30)
    print("[OK] FarField added:", ff.Label)
end)
if not ok then print("[FAIL] FarField:", err) end

-- 5. 添加近场请求（包围机体的笛卡尔网格）
ok, err = pcall(function()
    local config = project.Contents.SolutionConfigurations[1]
    local nf = config.NearFields:AddCartesian(-1.0, -0.8, -0.6, 1.5, 0.8, 0.6, 11, 9, 7)
    print("[OK] NearField added:", nf.Label)
end)
if not ok then print("[FAIL] NearField:", err) end

-- 6. 保存项目
ok, err = pcall(function()
    application:SaveAs([[f:\MyWorkSpace\UAVGame\3d_feko_run\test_connect.cfx]])
    print("[OK] Saved test_connect.cfx")
end)
if not ok then print("[FAIL] SaveAs:", err) end

-- 7. 运行 FEKO 求解（完整循环验证）
ok, err = pcall(function()
    local result = application.Launcher:RunFEKO()
    print("[OK] RunFEKO Succeeded =", result.Succeeded)
end)
if not ok then print("[FAIL] RunFEKO:", err) end

print("=== test_connect.lua 执行完毕 ===")
