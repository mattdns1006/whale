run = {}

function run.train(inputs,targets)

	--local outputs, dLdO, loss

	if initModel == nil then
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())
		initModel = true
		norm,sign= torch.norm,torch.sign
	end
	
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()
		outputs = model:forward(inputs) -- Only one input for training unlike testing
		loss = criterion:forward(outputs,targets)
		dLdO = criterion:backward(outputs,targets)

		model:backward(inputs,dLdO)
		return	loss, gradParameters 
	end


	optimMethod(feval,parameters,optimState)

	return outputs,loss

end

function run.test(inputs,targets)

	local outputs = model:forward(inputs)
	local loss = criterion:forward(outputs,targets)
	return outputs, loss
end

return run
