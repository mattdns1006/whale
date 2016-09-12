require "paths"
require "image"
require "cunn"

loadData = {}
loadData.__index = loadData

function loadData.init(tid,nThreads,batchSize) 
	local self = {}
	self.tid = tid
	self.bs = batchSize
	self.epoch = 1
	self.xPaths = {}
	self.idx = 1 --Starting point

	local xPaths = {}
	for f in paths.files("augmented/","x_") do 
		xPaths[#xPaths + 1] = f
	end

	for i = self.tid, #xPaths, nThreads do 
		self.xPaths[#self.xPaths + 1] = xPaths[i]
	end

	self.xDims = image.loadPNG("augmented/"..self.xPaths[1]):size() -- Dimensions of x

	return setmetatable(self,loadData)
end

function loadData:getNextBatch()
	local start = self.idx
	X = {}
	Y = {}
	local path
	local counter = 1
	while true do
		path = self.xPaths[self.idx]
		table.insert(X, image.loadPNG("augmented/"..path):resize(1,3,self.xDims[2],self.xDims[3]))
		table.insert(Y, image.loadPNG("augmented/"..path:gsub("x_","y_")):resize(1,3,self.xDims[2],self.xDims[3]))
		if self.idx == #self.xPaths then self.idx = 1; self.epoch = self.epoch + 1; else self.idx = self.idx + 1 end
		if counter == self.bs then break else counter = counter + 1 end
	end
	self.X = torch.cat(X,1)
	self.Y = torch.cat(Y,1)
	return self.X, self.Y
end

function loadData:displayBatch()
end


function testLoadData()
	bs = 6
	nThreads = 60
	eg1 = loadData.init(1,nThreads,bs)
	eg2 = loadData.init(2,nThreads,bs)
	for i = 1, 30 do 
		eg1:getNextBatch()
		image.display(eg1.X)
		sys.sleep(0.2)
		print(i,eg1.idx,#eg1.xPaths,eg1.epoch)
	end
end



