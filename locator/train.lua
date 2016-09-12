function train(inputs,targets)

	local output
	local dLdO
	local loss

	if i == 1 then
		if model then parameters,gradParameters = model:getParameters() end
		print("Number of parameters ==>")
		print(parameters:size())
	end
	
	function feval(x)
		if x ~= parameters then parameters:copy(x) end
		gradParameters:zero()
		output = model:forward(inputs) -- Only one input for training unlike testing
		--dLdO 
		model:backward(inputs,dLdO)

		return	loss, gradParameters 
	end

	_, loss = optimMethod(feval,parameters,optimState)

	if i % params.lrChange == 0 then
		local clr = params.lr
		params.lr = params.lr/params.lrDecay
		print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
			optimState = {
				learningRate = params.lr,
				beta1 = 0.9,
				beta2 = 0.999,
				epsilon = 1e-8
			}
	end
	if i % params.modelSave == 0 then
		print("==> Saving model " .. modelName .. ".")
		torch.save(modelName,model)
	end
	xlua.progress(i,params.nIter)
	i = i + 1
	return output,targetResize,loss

end

function test(inputs,target)

	local output
	local loss
	local targetResize

	output = model:forward(inputs)
	loss = criterion:forward(output,target)
	return output, targetResize, loss
end
