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

cmd:option("-inW",400,"Input size")
cmd:option("-inH",300,"Input size")
cmd:option("-sf",0.7,"Scaling factor.")
cmd:option("-nFeats",16,"Number of features.")
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
cmd:option("-saveTest",0,"Save test.")

cmd:option("-nDown",4,"Number of down steps.")
cmd:option("-nUp",1,"Number of up steps.")
cmd:option("-dsFactor",2,"Reduce image by factor.")

cmd:option("-outH",14,"Number of down steps.")
cmd:option("-outW",20,"Number of up steps.")
cmd:text()

params = cmd:parse(arg)
models = require "models"
optimState = {learningRate = params.lr, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8 }
optimMethod = optim.adam

print("Model name ==>")
modelName = "deconv3.model"
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
				       local outputs, dstPath
					if params.test == 1 then
						model:evaluate()
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
					       model:training()
					       outputs, loss = train(X,Y)
					       dScore = diceScore(outputs,Y)
					       display(X,Y,outputs,"train",4,5) 
					       i = i + 1
					       table.insert(losses, loss)
					       table.insert(dScores, dScore)
					       if i % 400 ==0 then
						       local lT =  torch.Tensor(losses)
						       local dST =  torch.Tensor(dScores)

						       print(string.format("Mean loss and dice score = %f , %f. \n",lT:mean(),dST:mean()))
						       losses = {}
						       dScores = {}
						       --local t  =  torch.range(1,#losses)
						       --gnuplot.plot({t,lT},{t,dST})
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

	if params.display == 1 then
		if imgDisplay == nil then 
			local initPic = torch.range(1,torch.pow(100,2),1):reshape(100,100)
			imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
		end
	end
	assert(#pathsToFit <= 11468,"Number to fit /< 11468")
	for i = 1, #pathsToFit do 
		x,name = feed:getNextBatch("test")
		o = model:forward(x):squeeze()
		o = image.scale(o:double(),params.inW,params.inH)

		level = params.toFitLevel
		dstName = name[1]:gsub(strMatch,"m"..tostring(level)) -- m for mask!

		image.save(dstName,o)


		if i % 25 == 0 then 
			xlua.progress(i,#pathsToFit)
			
			print(level,name[1],dstName)
		end
		collectgarbage()
	end
	print(string.format("Time taken = %f seconds ", timer:time().real))

end

