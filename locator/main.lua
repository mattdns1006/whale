require "image"
require "gnuplot"
require "nn"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"
local fnsPath = "/Users/matt/torchFunctions/"
fns = {}
table.insert(fns,"deconvDisplay.lua"); table.insert(fns,"layers.lua"); table.insert(fns,"csv.lua"); table.insert(fns,"shuffle.lua"); table.insert(fns,"diceScore.lua");
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

cmd:option("-bs",3,"Batch size.")
cmd:option("-lr",0.001,"Learning rate.")
cmd:option("-lrDecay",1.2,"Learning rate change factor.")
cmd:option("-lrChange",10000,"How often to change lr.")

cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",10000,"Number of iterations.")
cmd:option("-zoom",3,"Image zoom.")

cmd:option("-ma",100,"Moving average.")
cmd:option("-run",1,"Run.")
cmd:option("-modelSave",1000,"Model save frequency.")
cmd:option("-test",0,"Test mode.")

cmd:option("-nDown",8,"Number of down steps.")
cmd:option("-nUp",2,"Number of up steps.")

cmd:option("-outH",14,"Number of down steps.")
cmd:option("-outW",20,"Number of up steps.")
cmd:text()

params = cmd:parse(arg)
models = require "models"
optimState = {learningRate = params.lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 }
optimMethod = optim.adam

print("Model name ==>")
modelName = "deconv.model"
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
					if params.test == 1 then
						outputs = model:forward(X)
						for i = 1, outputs:size(1) do 
							dstPath = Y[i]:gsub("w_","lf_")
							image.saveJPG(dstPath,outputs[i])
						end
						i = i + 1 
						if i % 50 == 0 then 
							xlua.progress(i,12007)
						end
						--display(X,Y,outputs,"test",3,10)

					else 
					       outputs, loss = train(X,Y)
					       dScore = diceScore(outputs,Y)
					       display(X,Y,outputs,"train",2,30) 
					       i = i + 1
					       table.insert(losses, loss)
					       table.insert(dScores, dScore)
					       if i % 20 ==0 then
						       local lT =  torch.Tensor(losses)
						       local dST =  torch.Tensor(dScores)
						       local t  =  torch.range(1,#losses)
						       gnuplot.plot({t,lT},{t,dST})
						        --collectgarbage()
						end
						xlua.progress(i,params.nIter)

						if i % params.lrChange == 0 then
							local clr = params.lr
							params.lr = params.lr/params.lrDecay
							print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
							learningRate = params.lr
						end
						if i % params.modelSave == 0 then
							print("==> Saving model " .. modelName .. ".")
							torch.save(modelName,model)
						end

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
	for i = 1, #pathsToFit do 
		x,name = feed:getNextBatch("test")
		o = model:forward(x)
		dstPath = name[1]:gsub("wS_","lf_")
		image.save(dstPath,o[1])
		if i % 50 == 0 then 
			xlua.progress(i,#pathsToFit)
		end
		collectgarbage()
	end
	print(string.format("Time taken = %f seconds ", timer:time().real))

end

