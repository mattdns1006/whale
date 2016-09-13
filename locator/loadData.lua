require "paths"
require "image"
require "cunn"
dofile("/Users/matt/torchFunctions/shuffle.lua")

loadData = {}
loadData.__index = loadData

pathsToFit = {}

dir = "../imgs"
for d in paths.iterdirs(dir) do
	local d = dir .. "/" .. d .. "/"
	for f in paths.files(d,"wS") do
		f = d .. f
		pathsToFit[#pathsToFit + 1] = f
	end
end

function loadData.init(tid,nThreads,batchSize) 
	local self = {}
	self.tid = tid
	self.bs = batchSize
	self.epoch = 1
	self.xPaths = {}
	self.pathsToFit = {}
	self.finishedTest = 0
	self.idx = 1 --Starting point

	local xPaths = {}
	for f in paths.files("augmented/","x_") do 
		xPaths[#xPaths + 1] = f
	end

	for i = self.tid, #xPaths, nThreads do 
		self.xPaths[#self.xPaths + 1] = xPaths[i]
	end

	for i = self.tid, #pathsToFit, nThreads do 
		self.pathsToFit[#self.pathsToFit + 1] = pathsToFit[i]
	end

	self.xDims = image.loadPNG("augmented/"..self.xPaths[1]):size() -- Dimensions of x
	self.xPaths = shuffle(self.xPaths)

	return setmetatable(self,loadData)
end

function loadData:getNextBatch(trainOrTest)
	local X = {}
	local Y = {}
	local path
	local counter = 1
	local x, y, inW, inH 
	local dsFactor = 2

	if trainOrTest == "train" then
		while true do
			path = self.xPaths[self.idx]
			x = image.loadPNG("augmented/"..path)
			inW, inH = self.xDims[3]/dsFactor, self.xDims[2]/dsFactor
			x = image.scale(x,inW,inH,"bilinear"):resize(1,3,inH,inW)
			table.insert(X, x)
			y = image.loadPNG("augmented/"..path:gsub("x_","y_"),1)
			y = image.scale(y,params.outW,params.outH,"bilinear"):resize(1,1,params.outH,params.outW)
			table.insert(Y,y) 
			if self.idx == #self.xPaths then self.idx = 1; self.epoch = self.epoch + 1; self.xPaths = shuffle(self.xPaths); else self.idx = self.idx + 1 end
			if counter == self.bs then break else counter = counter + 1 end
		end
		self.X = torch.cat(X,1):cuda()
		self.Y = torch.cat(Y,1):cuda()
		collectgarbage()
		return self.X, self.Y
	elseif trainOrTest == "test" then
		if self.finishedTest == 1 then
			print(tid, " is sleeping.")
			sys.sleep(10)
		else 
			local names = {}
			while true do
				path = self.pathsToFit[self.idx]
				names[#names + 1] = path
				x = image.loadJPG(path)
				inW, inH = self.xDims[3]/dsFactor, self.xDims[2]/dsFactor
				x = image.scale(x,inW,inH,"bilinear"):resize(1,3,inH,inW)
				table.insert(X, x)
				if self.idx == #self.pathsToFit then self.finishedTest = 1 else; self.idx = self.idx + 1 end
				if counter == self.bs then break else counter = counter + 1 end
			end

			self.X = torch.cat(X,1):cuda()
			collectgarbage()
			return self.X, names
		end
	end
end

function testLoadData()
	bs = 6
	nThreads = 60
	dofile("/Users/matt/torchFunctions/shuffle.lua")
	params = {}
	params.outW = 40
	params.outH = 32
	eg1 = loadData.init(1,nThreads,bs)
	eg2 = loadData.init(2,nThreads,bs)
	for i = 1, 5 do 
		eg1:getNextBatch()
		image.display(eg1.X)
		sys.sleep(0.2)
		print(i,eg1.idx,#eg1.xPaths,eg1.epoch)
	end
end





