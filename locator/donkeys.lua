Threads = require "threads"

do 
	local threadParams = params
	donkeys = Threads(
			params.nThreads,
			function(idx)
				params = threadParams
				require "xlua"
				require "string"
				tid = idx -- Thread id
				dofile("loadData.lua")
				dofile("/Users/matt/torchFunctions/shuffle.lua")
				dataFeed = loadData.init(tid,params.nThreads,params.bs)
				print(string.format("Initialized thread %d.",tid))
			end
			)
end
