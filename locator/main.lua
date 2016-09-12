require "image"
require "gnuplot"
require "nn"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"
local fnsPath = "/Users/matt/torchFunctions/"
fns = {}
table.insert(fns,"deconvDisplay.lua"); table.insert(fns,"layers.lua"); table.insert(fns,"csv.lua"); table.insert(fns,"shuffle.lua");
for k,v in ipairs(fns) do; dofile(fnsPath..v) end
dofile("train.lua")


cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","locator.model","Name of model.")
cmd:option("-modelSave",5000,"How often to save.")
cmd:option("-loadModel",0,"Load model.")
cmd:option("-nThreads",3,"Number of threads.")
cmd:option("-trainAll",0,"Train on all images in training set.")
cmd:option("-actualTest",0,"Acutal test predictions.")

cmd:option("-inW",290,"Input size")
cmd:option("-inH",210,"Input size")
cmd:option("-sf",0.7,"Scaling factor.")
cmd:option("-nFeats",22,"Number of features.")
cmd:option("-kernelSize",3,"Kernel size.")

cmd:option("-bs",5,"Batch size.")
cmd:option("-lr",0.001,"Learning rate.")
cmd:option("-lrDecay",1.2,"Learning rate change factor.")
cmd:option("-lrChange",10000,"How often to change lr.")

cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",2000000,"Number of iterations.")
cmd:option("-zoom",3,"Image zoom.")

cmd:option("-ma",100,"Moving average.")
cmd:option("-run",1,"Run.")

cmd:option("-nDown",8,"Number of down steps.")
cmd:option("-nUp",2,"Number of up steps.")
cmd:text()

params = cmd:parse(arg)
models = require "models"
optimState = { learningRate = params.lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 }

optimMethod = optim.adam

print("Model name ==>")
modelName = string.format("models/deconv_%d_%d_%d_%d_%d_%d",params.inH,params.inW,params.nFeats,params.nDown,params.nUp,params.kernelSize)
if params.loadModel == 1 then
	print("==> Loading model")
	model = torch.load(modelName):cuda()
else 	
	model = models.model1():cuda()
end
criterion = nn.MSECriterion():cuda()
print("==> Init threads")
dofile("donkeys.lua")

function run()
	i = 1
	while i < 200 do
		donkeys:addjob(function()
					X, Y = dataFeed:getNextBatch()
					return X,Y
			       end,
			       function(X,Y)
				      
			       end
			     )
		i = i + 1
	end
end

if params.run == 1 then run() end
	
