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
	layers.init(model)
	return model
end



return models
