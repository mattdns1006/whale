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

function imageCentroid(img)
	assert(img:nDimension() == 2,"needs to be of dim 2")
	local nH, nW = img:size(1), img:size(2)
	local sumH, sumW, sum = img:sum(2), img:sum(1), img:sum()
	local h, w = torch.range(1,nH), torch.range(1,nW)
	local mH,mW = torch.cmul(h,sumH):sum()/sum, torch.cmul(w,sumW):sum()/sum
	return mW, mH
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
	local counter = 1
	local x, y, inW, inH, mask, path

	if trainOrTest == "train" then
		while true do
			path = self.xPaths[self.idx]
			x = image.loadPNG("augmented/"..path)
			--x = x:csub(x:mean())
			inW, inH = params.inW, params.inH 
			x = image.scale(x,inW,inH,"bilinear"):resize(1,3,inH,inW)
			table.insert(X, x)
			mask = image.loadPNG("augmented/"..path:gsub("x_","y_"),1)
			mask = image.scale(mask,inW,inH,"bilinear")
			mx,my = imageCentroid(mask)
			mx = mx/inW  -- normalize
			my = my/inH  -- normalize
			y = torch.Tensor{mx,my}:resize(1,2)
			table.insert(Y, y)

			
			if self.idx == #self.xPaths then self.idx = 1; self.epoch = self.epoch + 1; self.xPaths = shuffle(self.xPaths); else self.idx = self.idx + 1 end
			if counter == self.bs then break else counter = counter + 1 end

		end
		self.X = torch.cat(X,1):cuda()
		self.Y = torch.cat(Y,1):cuda()
		collectgarbage()
		return self.X, self.Y
	end
end

function testLoadData()
	bs = 10
	nThreads = 100
	dofile("/Users/matt/torchFunctions/shuffle.lua")
	dis = image.display{image=torch.rand(100,100),zoom=5,offscreen=false}
	
	params = {}
	params.outW = 40
	params.outH = 32
	params.inW = 150
	params.inH = 100
	params.dsFactor = 1
	eg = loadData.init(1,nThreads,bs)
	while true do
		x,y = eg:getNextBatch("train")
		image.display{image = x, win = dis}
		print(eg.idx)
	end

	--[[
	while true do
		x,y = eg:getNextBatch("train")
		img = x[1]:double()
		coords = y[1]:double()
		coords[1] = coords[1]*params.inW
		coords[2] = coords[2]*params.inH
		print(coords)

		mx,my =  coords[1], coords[2]
		mx,my = math.floor(mx),math.floor(my)

		if my - 4 < 1 or my + 4 >= img:size(2) or mx - 4 < 1 or mx + 4 > img:size(3) then
			else
		img:narrow(2,my-4,8):narrow(3,mx-4,8):fill(0)
		image.display{image = img, win = dis}
	end
	]]--
end





