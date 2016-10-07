require "image"
require "gnuplot"
require "nn"
require "cunn"
require "xlua"
require "optim"
require "gnuplot"
local fnsPath = "/home/msmith/misc/torchFunctions/"
fns = {}
table.insert(fns,"deconvDisplay.lua"); table.insert(fns,"layers.lua"); table.insert(fns,"csv.lua"); table.insert(fns,"shuffle.lua"); table.insert(fns,"diceScore.lua");
dofile("/home/msmith/misc/torchFunctions/shuffle.lua")
for k,v in ipairs(fns) do; dofile(fnsPath..v) end

dofile("train.lua")


cmd = torch.CmdLine()
cmd:text()
cmd:text("Options")
cmd:option("-modelName","locator.model","Name of model.")
cmd:option("-modelSave",5000,"How often to save.")
cmd:option("-loadModel",1,"Load model.")
cmd:option("-nThreads",8,"Number of threads.")
cmd:option("-trainAll",0,"Train on all images in training set.")
cmd:option("-actualTest",0,"Acutal test predictions.")

cmd:option("-inW",900,"Input size")
cmd:option("-inH",600,"Input size")
cmd:option("-sf",0.7,"Scaling factor.")
cmd:option("-nFeats",32,"Number of features.")
cmd:option("-featInc",32,"Number of features increasing.")
cmd:option("-kernelSize",3,"Kernel size.")

cmd:option("-bs",3,"Batch size.")
cmd:option("-lr",0.0001,"Learning rate.")
cmd:option("-lrDecay",1.1,"Learning rate change factor.")
cmd:option("-lrChange",10000,"How often to change lr.")

cmd:option("-display",0,"Display images.")
cmd:option("-displayFreq",100,"Display images frequency.")
cmd:option("-displayGraph",0,"Display graph of loss.")
cmd:option("-displayGraphFreq",500,"Display graph of loss.")
cmd:option("-nIter",1000000,"Number of iterations.")
cmd:option("-zoom",3,"Image zoom.")

cmd:option("-ma",100,"Moving average.")
cmd:option("-run",1,"Run.")
cmd:option("-modelSave",5000,"Model save frequency.")
cmd:option("-toFitLevel",1,"Fitting (test mode) which level eg. 1 2 or 3.")
cmd:option("-test",0,"Test mode.")
cmd:option("-testSave",0,"Save test.")

cmd:option("-nDown",5,"Number of down steps.")
cmd:option("-nUp",1,"Number of up steps.")

cmd:option("-outH",14,"Number of down steps.")
cmd:option("-outW",20,"Number of up steps.")
cmd:text()

params = cmd:parse(arg)
models = require "models"
optimState = {learningRate = params.lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 }
optimMethod = optim.adam
logger = optim.Logger("model.log")


print("Model name ==>")
modelName = "deconv4.model"
if params.loadModel == 1 then
	print("==> Loading model")
	model = torch.load(modelName):cuda()
else 	
	model = models.model1():cuda()
end
testInput = torch.randn(1,3,params.inH,params.inW):cuda()
outSize = model:forward(testInput):size()
print("Output size ==> ", outSize)
params.outH = outSize[3]
params.outW = outSize[4]
criterion = nn.MSECriterion():cuda()

logger:add(params)


dofile("loadData.lua")
feed = loadData.init(1,1,1)

function run()
	print("==> Init threads")
	dofile("donkeys.lua")
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

					-- Running test views

				       if i % 50 == 0 then 
						if params.display == 1 then
							if testDisplay == nil then 
								local initPic = torch.range(1,torch.pow(100,2),1):reshape(100,100)
								test1 = image.display{image=initPic, zoom=zoom, offscreen=false}
								test2 = image.display{image=initPic, zoom=zoom, offscreen=false}
								testDisplay = 1
							end
							x,name = feed:getNextBatch("test")
							model:evaluate()
							o = model:forward(x):squeeze()
							image.display{image = x, win = test1, title = name} 
							image.display{image = o, win = test2, title = name} 
						end
					end

				       model:training()
				       outputs, loss = train(X,Y)
				       dScore = diceScore(outputs,Y)
				       display(X,Y,outputs,"train",4,5) 

				       table.insert(losses, loss)
				       table.insert(dScores, dScore)

				       if i % 400 ==0 then
					       -- Print moving average scores for training set
					       local lT =  torch.Tensor(losses)
					       local dST =  torch.Tensor(dScores)
					       print(string.format("Mean loss and dice score = %f , %f. \n",lT:mean(),dST:mean()))
					       losses = {}
					       dScores = {}
					end


					if i % params.lrChange == 0 then
						local clr = params.lr
						params.lr = params.lr/params.lrDecay
						print(string.format("Learning rate dropping from %f ====== > %f. ",clr,params.lr))
						learningRate = params.lr
					end
					if i % params.modelSave == 0 then print("==> Saving model " .. modelName .. ".") torch.save(modelName,model) end

				        i = i + 1
					xlua.progress(i,params.nIter)

					collectgarbage()
				      
			       end
			     )
	end
end

function fitMasks()

	model:evaluate()

	if params.display == 1 then
		if imgDisplay == nil then 
			local initPic = torch.range(1,torch.pow(100,2),1):reshape(100,100)
			imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
		end
	end

	for i = 1, #pathsToFit do 
		x,name = feed:getNextBatch("test")
		dstName = name[1]:gsub(strMatch,"m"..tostring(level)) -- m for mask!
		o = model:forward(x):squeeze()
		image.display{image = o, win = imgDisplay0, title = dstName} 
		o = image.scale(o:double(),params.inW,params.inH)

		level = params.toFitLevel

		if params.testSave == 1 then image.save(dstName,o) end
		if params.display == 1 then 
			image.display{image = o, win = imgDisplay0, title = dstName} 
			sys.sleep(1)
		end
		
		if i % 25 == 0 then 
			xlua.progress(i,#pathsToFit)
			print(level,name[1],dstName)
		end

		collectgarbage()
	end
end

if params.run == 1 and params.test ==0 then run() end
if params.test == 1 then fitMasks() end

