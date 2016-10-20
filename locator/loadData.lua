require "paths"
require "image"
require "cunn"
dofile("/home/msmith/misc/torchFunctions/shuffle.lua")


loadData = {}
loadData.__index = loadData

local pathsToFit = {} -- To fit (everything)
local xPaths = {} -- Train
if params then

	for f in paths.files("augmented/","x_") do 
		xPaths[#xPaths + 1] = f
	end

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
	self.testToFit = {}
	self.finishedTest = 0
	self.xPathsIdx = 1 --Starting point
	self.pathsToFitIdx = 1 --Starting point


	for i = self.tid, #xPaths, nThreads do 
		self.xPaths[#self.xPaths + 1] = xPaths[i]
	end

	for i = self.tid, #pathsToFit, nThreads do 
		self.pathsToFit[#self.pathsToFit + 1] = pathsToFit[i]
	end

	--[[
	for i = self.tid, #testToFit , nThreads do 
		self.testToFit[#self.testToFit + 1] = testToFit[i]
	end
	]]--

	self.xDims = image.loadJPG("augmented/"..self.xPaths[1]):size() -- Dimensions of x
	self.xPaths = shuffle(self.xPaths)
	self.pathsToFit= shuffle(self.pathsToFit)

	return setmetatable(self,loadData)
end



function loadData:getNextBatch(trainOrTest)
	local X = {}
	local Y = {}
	local path
	local counter = 1
	local x, y, inW, inH 
	local inW, inH = params.inW, params.inH 
	local outW, outH = params.outH,params.outW
	local aug, fit, t
	if trainOrTest == "train" then
		fit = 0 
		t = self.xPaths
		pathStart = "augmented/"
	else if trainOrTest == "test" then
		t = self.pathsToFit
		fit = 1
		pathStart = ""
	end
	while true do

	end
end

function augment(x,y)

	--[[
	local randomRotate = torch.uniform(-5,5)
	local x = image.rotate(x,randomRotate,"bilinear")
	local y = image.rotate(y,randomRotate,"bilinear")
	]]--

	local c,h,w = x:size(2), x:size(3), x:size(4)
	local ar = w/h
	local y = image.scale(y,w,h,"bilinear") -- make sure same size before cropping


	local maxCrop = torch.floor(0.1*w)
	local xCrop = torch.random(3,maxCrop)
	local yCrop = torch.floor(xCrop/ar)

	local x = x:narrow(4,xCrop,w-xCrop-1)
	local x = x:narrow(3,yCrop,h-yCrop-1)

	local y = y:narrow(2,xCrop,w-xCrop-1)
	local y = y:narrow(1,yCrop,h-yCrop-1)

	
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
	--img()

	bs = 10
	nThreads = 1
	dofile("/home/msmith/misc/torchFunctions/shuffle.lua")
	params = {}
	params.inW = 320
	params.inH = 200
	params.outW = 40
	params.outH = 32
	eg1 = loadData.init(1,nThreads,bs)
	eg2 = loadData.init(2,nThreads,bs)
	for i = 1, 5 do 
		x, o = eg1:getNextBatch("train")
		--image.display{image = o, win = imgDisplay0, title = dstName} 
		sys.sleep(0.2)
		print(i,eg1.idx,#eg1.xPaths,eg1.epoch)
	end
end





