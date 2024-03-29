require "nn"

local Convolution = nn.SpatialConvolution
local Pool = nn.SpatialMaxPooling
local fmp = nn.SpatialFractionalMaxPooling
local UpSample = nn.SpatialUpSamplingNearest
local SBN = nn.SpatialBatchNormalization
local af = nn.ReLU
local Linear = nn.Linear
local Dropout = nn.Dropout

models = {}

function initParamsEg()
	params = {}
	params.kernelSize = 3
	params.nFeats = 22
	params.nDown = 7
	params.nUp = 3 
	model = nn.Sequential()
end

local nFeats = params.nFeats 
local nFeatsInc = torch.floor(params.nFeats/4)
local nOutputs
local nInputs
local kS = params.kernelSize
local pad = torch.floor((kS-1)/2)

function shortcut(nInputPlane, nOutputPlane, stride)
	return nn.Sequential()
		:add(Convolution(nInputPlane,nOutputPlane,1,1,stride,stride,0,0))
		:add(SBN(nOutputPlane))
end
	
function basicblock(nInputPlane, n, stride)
	local s = nn.Sequential()

	s:add(Convolution(nInputPlane,n,3,3,1,1,1,1))
	s:add(SBN(n))
	s:add(af())

	return nn.Sequential()
	 :add(nn.ConcatTable()
	    :add(s)
	    :add(shortcut(nInputPlane, n, stride)))
	 :add(nn.CAddTable(true))
	 :add(af())

end

function block(model,nInputs,nOutputs)
	model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
	model:add(SBN(nInputs))
	model:add(af())

end

function models.model1() 
	local model = nn.Sequential()
	local featInc = params.featInc 
	local nInputs =  params.nFeats
	local nOutputs = nInputs + featInc

	for i = 1, params.nDown do
		if i == 1 then nInputs = 3 end
		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		model:add(SBN(nOutputs))
		model:add(af())

		--[[
		model:add(nn.Dropout(0.1))
		model:add(Convolution(nOutputs,nOutputs,3,3,1,1,1,1))
		model:add(SBN(nOutputs))
		model:add(af())
		
		]]--
		model:add(Pool(2,2,2,2,0,0))
		
		nInputs = nOutputs
		nOutputs = nOutputs + featInc
	end

	print("Size at lowest spatial size ==> ", model:cuda():forward(torch.rand(1,3,params.inH,params.inW):cuda()):size())
	

	for i = 1, params.nUp do 

		model:add(Convolution(nInputs,nOutputs,3,3,1,1,1,1))
		model:add(SBN(nOutputs))
		model:add(af())
		model:add(UpSample(2))
		nInputs = nOutputs
		--nOutputs = nOutputs - featInc 
	end


	model:add(Convolution(nInputs,3,3,3,1,1,1,1))
	model:add(SBN(3))
	model:add(nn.Sigmoid())

	layers.init(model)



	return model
end



return models
