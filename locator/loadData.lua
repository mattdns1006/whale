require "paths"
require "image"
require "cunn"
dofile("/home/msmith/misc/torchFunctions/shuffle.lua")


loadData = {}
loadData.__index = loadData

pathsToFit = {}

if params then
	dir = "../imgs"
	for d in paths.iterdirs(dir) do
		local d = dir .. "/" .. d .. "/"
		if params.toFitLevel == 1 then  
			strMatch = "w"..tostring(params.toFitLevel).."S"
		else 
			strMatch = "w"..tostring(params.toFitLevel)
		end
		for f in paths.files(d,strMatch) do
			f = d .. f
			pathsToFit[#pathsToFit + 1] = f
		end
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
	local inW, inH = params.inW, params.inH 

	if trainOrTest == "train" then
		while true do
			path = self.xPaths[self.idx]

			x = image.loadPNG("augmented/"..path)
			y = image.loadPNG("augmented/"..path:gsub("x_","y_"))
			x,y = augment(x,y)

			x = image.scale(x,inW,inH,"bilinear"):resize(1,3,inH,inW)
			y = image.scale(y,params.outW,params.outH,"bilinear"):resize(1,3,params.outH,params.outW)

			table.insert(X, x)
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

function augment(x,y)

	local randomRotate = torch.uniform(-100,100)
	local x = image.rotate(x,randomRotate)
	local y = image.rotate(y,randomRotate)

	local c,h,w = x:size(1), x:size(2), x:size(3)
	local ar = w/h
	local y = image.scale(y,w,h,"bilinear") -- make sure same size before cropping

	local maxCrop = torch.floor(0.2*w)
	local xCrop = torch.random(3,maxCrop)
	local yCrop = torch.floor(xCrop/ar)

	local x = x:narrow(3,xCrop,w-xCrop-1)
	local x = x:narrow(2,yCrop,h-yCrop-1)

	local y = y:narrow(3,xCrop,w-xCrop-1)
	local y = y:narrow(2,yCrop,h-yCrop-1)

	
	return x,y
end

-- Testing 

function img()
	if imgDisplay == nil then 
		local initPic = torch.range(1,torch.pow(100,2),1):reshape(100,100)
		imgDisplay0 = image.display{image=initPic, zoom=zoom, offscreen=false}
	end
end

function testAugment()
	xPaths = {}
	for f in paths.files("augmented/","x_") do 
		xPaths[#xPaths + 1] = f
	end
	img()
	for i =1, 1000 do 
		path = xPaths[1]
		x = image.loadPNG("augmented/"..path)
		y = image.loadPNG("augmented/"..path:gsub("x_","y_"))
		x,y = augment(x,y)
		o = torch.cat(x,y)
		image.display{image = o, win = imgDisplay0} 
		sys.sleep(0.2)

	end
end

function testLoadData()
	img()

	bs = 10
	nThreads = 1
	dofile("/home/msmith/misc/torchFunctions/shuffle.lua")
	params = {}
	params.outW = 40
	params.outH = 32
	eg1 = loadData.init(1,nThreads,bs)
	eg2 = loadData.init(2,nThreads,bs)
	for i = 1, 5 do 
		eg1:getNextBatch()
		image.display{image = o, win = imgDisplay0, title = dstName} 
		sys.sleep(0.2)
		print(i,eg1.idx,#eg1.xPaths,eg1.epoch)
	end
end





