require "image"
require "cunn"
local fnsPath = "/home/msmith/misc/torchFunctions/"
fns = {}
table.insert(fns,"csv.lua"); table.insert(fns,"shuffle.lua"); table.insert(fns,"diceScore.lua");
dofile("/home/msmith/misc/torchFunctions/shuffle.lua")
for k,v in ipairs(fns) do; dofile(fnsPath..v) end

local fp = "../../"
local trainCV, testCV = csv.read(fp.."trainCV.csv",1), csv.read(fp.."testCV.csv",1)
weights = csv.read(fp.."trWeights.csv")
nClasses = 10 

provider = {}
provider.__index = provider

function provider.init(batchSize,inH,inW,c)
	self = {}
	self.batchSize = batchSize

	self.inH = inH 
	self.inW = inW 
	self.c = c 

	self.train = {}
	self.train.data =  trainCV
	self.train.nObs =  #trainCV
	self.train.idx = 1
	self.train.finishedEpoch = false
	self.train.weights = weights
	
	self.test = {}
	self.test.data = testCV
	self.test.nObs = #testCV
	self.test.idx = 1
	self.test.finishedEpoch = false

	return setmetatable(self,provider)
end

function provider:getNextBatch(trainOrTest)
	if trainOrTest == "train" then t = self.train else t = self.test end
	local to = math.min(t.nObs,self.batchSize+t.idx -1)
	
	local X = {}
	local Y = {}
	for i = t.idx, to do 
		obs = t.data[i]:split(",")
		label = tonumber(obs[4])
		X[#X+1] = image.loadJPG(fp.."imgs/"..obs[3].."/"..obs[2]:gsub("w_","head_")):resize(1,self.c,self.inH,self.inW)
		Y[#Y+1] = label
		t.idx = t.idx + 1
	end
	X = torch.cat(X,1):cuda()
	Y = torch.Tensor(Y):cuda()
	if to == t.nObs then t.idx = 1; t.data = shuffle(t.data); t.finishedEpoch = true end
	return X,Y
end

