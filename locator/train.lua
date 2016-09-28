function train(inputs,targets)

	local outputs, dLdO, loss

	if i == 1 then
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())

	end
	
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()
		outputs = model:forward(inputs) -- Only one input for training unlike testing
		dLdO = criterion:backward(outputs,targets)
		loss = criterion:forward(outputs,targets)
		model:backward(inputs,dLdO)

		return	loss, gradParameters 
	end

	optimMethod(feval,parameters,optimState)


	return outputs,loss

end

function test(inputs,target)

	local output
	local loss
	local targetResize

	output = model:forward(inputs)
	loss = criterion:forward(output,target)
	return output, targetResize, loss
end
