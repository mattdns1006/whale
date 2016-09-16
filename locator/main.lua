require "image"
require "gnuplot"
require "nn"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"
local fnsPath = "/Users/matt/torchFunctions/"
fns = {}
table.insert(fns,"layers.lua"); table.insert(fns,"csv.lua"); table.insert(fns,"shuffle.lua"); table.insert(fns,"diceScore.lua");
for k,v in ipairs(fns) do; dofile(fnsPath..v) end
dofile("train.lua")
dofile("display.lua")


cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","locator.model","Name of model.")
cmd:option("-modelSave",5000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nThreads",8,"Number of threads.")
cmd:option("-trainAll",0,"Train on all images in training set.")
cmd:option("-actualTest",0,"Acutal test predictions.")

cmd:option("-nFeats",22,"Number of features.")
cmd:option("-kernelSize",3,"Kernel size.")

cmd:option("-bs",4,"Batch size.")
cmd:option("-lr",0.003,"Learning rate.")
cmd:option("-lrDecay",1.2,"Learning rate change factor.")
cmd:option("-lrChange",10000,"How often to change lr.")

cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",3,"Display images seconds frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",1000000,"Number of iterations.")
cmd:option("-zoom",3,"Image zoom.")

cmd:option("-ma",100,"Moving average.")
cmd:option("-run",1,"Run.")
cmd:option("-modelSave",1000,"Model save frequency.")
cmd:option("-test",0,"Test mode.")
cmd:option("-saveTest",0,"Save test.")

cmd:option("-nDown",5,"N blocks.")
cmd:option("-inW",150,"Input size.")
cmd:option("-inH",100,"Output size.")

cmd:text()

params = cmd:parse(arg)
models = require "models"
optimState = {learningRate = params.lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 }
optimMethod = optim.adam

print("Model name ==>")
modelName = "locator.model"
if params.loadModel == 1 then
	print("==> Loading model")
	model = torch.load(modelName):cuda()
else 	
	model = models.model1():cuda()
end
criterion = nn.AbsCriterion():cuda()
print("==> Init threads")
dofile("donkeys.lua")

function run()
	i = 1
	losses = {}
	dScores = {}

	while i < params.nIter do
		donkeys:addjob(function()
				        if params.test == 1 then 
						X, names = dataFeed:getNextBatch("test")
						Y = names
					else 
						X, Y = dataFeed:getNextBatch("train")
					end
					return X,Y
			       end,
			       function(X,Y)
				       local outputs, dstPath
					       model:training()
					       outputs, loss = train(X,Y)
					       outputs:select(2,1):mul(params.inW)
					       outputs:select(2,2):mul(params.inH)
					       Y:select(2,1):mul(params.inW)
					       Y:select(2,2):mul(params.inH)
					       display(X,Y,outputs,"train",5,params.displayFreq) 
					       i = i + 1
					       table.insert(losses, loss)
					       if i % 50 ==0 then
						       local lT =  torch.Tensor(losses)
						       print(string.format("Mean loss %f",lT:mean()))
						       losses = {}
						       xlua.progress(i,params.nIter)
						end

						--[[

						if i % params.lrChange == 0 then
							local clr = params.lr
							params.lr = params.lr/params.lrDecay
							print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
							learningRate = params.lr
						end
						]]--
						if i % params.modelSave == 0 then
							print("==> Saving model " .. modelName .. ".")
							torch.save(modelName,model)
						end


					  collectgarbage()
			         end
				      
			     )
	end
end

if params.run == 1 and params.test ==0 then run() end

if params.test == 1 then
	dofile("loadData.lua")
	feed = loadData.init(1,1,1)
	timer = torch.Timer()
	local x, o
	model:evaluate()
	for i = 1, #pathsToFit do 
		x,name = feed:getNextBatch("test")
		o = model:forward(x)
		dstPath = name[1]:gsub("wS_","lf_")
		if params.saveTest == 1 then
			image.save(dstPath,o[1])
		else 
			display(x,name,o,"test",4,0)
			sys.sleep(1)
		end

		if i % 50 == 0 then 
			xlua.progress(i,#pathsToFit)
		end
		collectgarbage()
	end
	print(string.format("Time taken = %f seconds ", timer:time().real))

end

