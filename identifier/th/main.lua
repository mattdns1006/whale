require "image"
require "gnuplot"
require "nn"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"
local fnsPath = "/home/msmith/misc/torchFunctions/"
fns = {}
table.insert(fns,"layers.lua"); table.insert(fns,"csv.lua"); table.insert(fns,"shuffle.lua"); table.insert(fns,"diceScore.lua"); table.insert(fns,"loss.lua")
dofile("/home/msmith/misc/torchFunctions/shuffle.lua")
for k,v in ipairs(fns) do; dofile(fnsPath..v) end

cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","identifier1.model","Name of model.")
cmd:option("-modelSave",5000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")

cmd:option("-batchSize",4,"duh")
cmd:option("-inH",420,"Input size (h).")
cmd:option("-inW",700,"Input size (w).")
cmd:option("-c",3,"Number of color channels.")

cmd:option("-nFeats",16,"Number of features.")
cmd:option("-nFeatsInc",16,"Number of features increasing.")
cmd:option("-kS",3,"Kernel size of convolutions.")
cmd:option("-nDown",7,"Number of blocks.")

cmd:option("-lr",0.0001,"Learning rate.")
cmd:option("-epochs",20,"N epochs")
cmd:text()


params = cmd:parse(arg)

optimState = {learningRate = params.lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 }
optimMethod = optim.adam
logger = optim.Logger("model.log")
logger:add(params)

dofile("loadData.lua")
models = require "models"
run = require "run"
criterion = nn.CrossEntropyCriterion(weightsT):cuda()

if params.loadModel == 1 then print("==> Loading model") model = torch.load(modelName):cuda() else model = models.model1():cuda() end

function runTrTe()
	feed = provider.init(params.batchSize,params.inH,params.inW,params.c)
	cmTr, cmTe = optim.ConfusionMatrix(nClasses), optim.ConfusionMatrix(nClasses)

	for epoch = 1, params.epochs do 
		trLoss = {} 
		teLoss = {}
		cmTr:zero()
		cmTe:zero()
		feed.train.finishedEpoch = false
		feed.test.finishedEpoch = false
		model:training()
		while feed.train.finishedEpoch == false do
			X,Y = feed:getNextBatch("train")
			outputs, loss = run.train(X,Y)

			if X:size(1) == 1 then
				cmTr:add(outputs,Y)
			else 
				cmTr:batchAdd(outputs,Y)
			end
			table.insert(trLoss,loss)
		end
		
		model:evaluate()
		while feed.test.finishedEpoch == false do
			X,Y = feed:getNextBatch("test")
			outputs, loss = run.test(X,Y)
			if X:size(1) == 1 then
				cmTe:add(outputs,Y)
			else 
				cmTe:batchAdd(outputs,Y)
			end
			table.insert(teLoss,loss)
		end
		print("Train CM")
		print(cmTr)
		print("Test CM")
		print(cmTe)
		print(string.format("Epoch %d == > Train loss = %f , test loss = %f",epoch,torch.Tensor(trLoss):mean(),torch.Tensor(teLoss):mean()))
		collectgarbage()
	end
end

--runTrTe()

