-- probe2.lua — 探查 NearField 实体实例的可设置属性名与设置方式
local log = io.open([[f:\MyWorkSpace\UAVGame\3d_feko_run\probe2.txt]], "w")

local function dump(tbl, prefix)
    prefix = prefix or ""
    if type(tbl) ~= "table" then
        log:write(prefix, tostring(tbl), "\n")
        return
    end
    local keys = {}
    for k in pairs(tbl) do keys[#keys + 1] = k end
    table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
    for _, k in ipairs(keys) do
        local v = tbl[k]
        if type(v) == "table" then
            log:write(prefix, k, ":\n")
            dump(v, prefix .. "  ")
        else
            log:write(prefix, k, " = ", tostring(v), "\n")
        end
    end
end

application = cf.Application.GetInstance()
project = application:NewProject()
local config = project.Contents.SolutionConfigurations[1]

local nf = config.NearFields:AddCartesian(-0.75, -0.75, -0.6, 1.21875, 0.71875, 0.36875, 4, 4, 4)
log:write("nf added\n")

local ok1, err1 = pcall(function()
    local props = nf:GetProperties()
    dump(props, "P.")
end)
if not ok1 then log:write("GetProperties FAIL: " .. tostring(err1) .. "\n") end

local ok2, err2 = pcall(function() nf:SetProperties({ CalculateMagneticFields = false }) end)
log:write("SetMagOff: " .. tostring(ok2) .. " err=" .. tostring(err2) .. "\n")

local ok3, err3 = pcall(function() nf:SetProperties({ OnlyScatteredPartCalculationEnabled = true }) end)
log:write("SetScat: " .. tostring(ok3) .. " err=" .. tostring(err3) .. "\n")

local ok4, err4 = pcall(function() nf.OnlyScatteredPartCalculationEnabled = true end)
log:write("DirectScat: " .. tostring(ok4) .. " err=" .. tostring(err4) .. "\n")

local ok5, err5 = pcall(function() nf.CalculateMagneticFields = false end)
log:write("DirectMagOff: " .. tostring(ok5) .. " err=" .. tostring(err5) .. "\n")

log:close()
print("probe2 done")
