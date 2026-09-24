-- probe3.lua — 验证 NearField.Advanced 嵌套属性设置
local log = io.open([[f:\MyWorkSpace\UAVGame\3d_feko_run\probe3.txt]], "w")
application = cf.Application.GetInstance()
project = application:NewProject()
local config = project.Contents.SolutionConfigurations[1]
local nf = config.NearFields:AddCartesian(-0.75, -0.75, -0.6, 1.21875, 0.71875, 0.36875, 4, 4, 4)

local ok, err = pcall(function()
    nf:SetProperties({
        Advanced = {
            CalculateMagneticFields = false,
            OnlyScatteredPartCalculationEnabled = true,
        }
    })
end)
log:write("nested set: " .. tostring(ok) .. " err=" .. tostring(err) .. "\n")

local props = nf:GetProperties()
log:write("after: CalculateMagneticFields = " .. tostring(props.Advanced.CalculateMagneticFields) .. "\n")
log:write("after: OnlyScatteredPartCalculationEnabled = " .. tostring(props.Advanced.OnlyScatteredPartCalculationEnabled) .. "\n")
log:close()
print("probe3 done")
