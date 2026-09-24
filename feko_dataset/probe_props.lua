-- probe_props.lua — 探查 FEKO Lua API 的 NearField / FarField 属性（供批量脚本使用）
local log = io.open([[f:\MyWorkSpace\UAVGame\3d_feko_run\probe_props.txt]], "w")

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

local ok1, err1 = pcall(function() dump(cf.NearField.GetDefaultProperties(), "NF.") end)
if not ok1 then log:write("ERR NearField: ", err1, "\n") end

local ok2, err2 = pcall(function() dump(cf.FarField.GetDefaultProperties(), "FF.") end)
if not ok2 then log:write("ERR FarField: ", err2, "\n") end

local ok3, err3 = pcall(function() dump(cf.PlaneWave.GetDefaultProperties(), "PW.") end)
if not ok3 then log:write("ERR PlaneWave: ", err3, "\n") end

log:close()
print("probe done")
