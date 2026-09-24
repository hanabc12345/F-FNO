-- =============================================================================
-- test_mesher.lua
-- FEKO Mesher 完整测试：导入修复网格 → 显式设置三角形边长 → 重划分 → 求解
-- =============================================================================
local LOG = [[f:\MyWorkSpace\UAVGame\3d_feko_run\test_mesher.log]]
local function log(msg)
    local f = io.open(LOG, "a")
    if f then f:write(msg .. "\n"); f:close() end
end

application = cf.Application.GetInstance()
project = application:NewProject()

local stl_path = [[f:\MyWorkSpace\UAVGame\3d_feko_run\f16_fixed.stl]]
local ok, err = pcall(function()
    local meshes = project.Importer.MeshImporter:Import(stl_path)
    log("Import STL, count = " .. #meshes)
end)
if not ok then log("[FAIL] Import: " .. tostring(err)) end

pcall(function()
    project.Contents.SolutionConfigurations.GlobalFrequency.Start = "3e9"
end)

-- 显式设置三角形目标边长（米）
ok, err = pcall(function()
    local s = project.Mesher.Settings
    s:SetProperties({
        ["MeshSizeOption"] = "Manual",   -- 尝试手动模式
        ["TriangleEdgeLength"] = "0.012",
    })
    log("Set TriangleEdgeLength=0.012, MeshSizeOption=Manual")
end)
if not ok then log("[FAIL] SetProperties Manual: " .. tostring(err)) end

-- 再读取确认
ok, err = pcall(function()
    local props = project.Mesher.Settings:GetProperties()
    log("  after set: MeshSizeOption = " .. tostring(props.MeshSizeOption))
    log("  after set: TriangleEdgeLength = " .. tostring(props.TriangleEdgeLength))
end)
if not ok then log("[FAIL] GetProperties2: " .. tostring(err)) end

-- 重划分
ok, err = pcall(function()
    project.Mesher:Mesh()
    log("Mesh() called")
end)
if not ok then log("[FAIL] Mesh(): " .. tostring(err)) end

-- 平面波 + 远场 + 近场
pcall(function()
    project.Contents.SolutionConfigurations.GlobalSources:AddPlaneWave(90, 0)
end)
pcall(function()
    local config = project.Contents.SolutionConfigurations[1]
    config.FarFields:Add(0, 0, 180, 360, 30, 30)
end)
pcall(function()
    local config = project.Contents.SolutionConfigurations[1]
    config.NearFields:AddCartesian(-1.0, -0.8, -0.6, 1.5, 0.8, 0.6, 11, 9, 7)
end)

ok, err = pcall(function()
    application:SaveAs([[f:\MyWorkSpace\UAVGame\3d_feko_run\test_mesher.cfx]])
    log("Saved test_mesher.cfx")
end)
if not ok then log("[FAIL] SaveAs: " .. tostring(err)) end

-- 求解
ok, err = pcall(function()
    local result = application.Launcher:RunFEKO()
    log("RunFEKO Succeeded = " .. tostring(result.Succeeded))
end)
if not ok then log("[FAIL] RunFEKO: " .. tostring(err)) end

log("=== test_mesher.lua 执行完毕 ===")
