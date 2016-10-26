import cv2, sys
import pandas as pd
import numpy as np
from random import shuffle

def normalize(img):
	img = img.astype(np.uint8)
	img = img/255.0
	return img

def feed(inDims, outDims, paths):
	
	paths = paths
	nObs = len(paths)
	X = np.zeros(inDims).astype(np.float32)
	Y = np.zeros(outDims).astype(np.float32)
	batchSize = inDims[0]
	pathIdx = 0

	while True:

		for i in range(batchSize):
			path = paths[pathIdx]
			x = cv2.imread(path)
			y = cv2.imread(path.replace("x_","y_"))

			x = cv2.resize(x,(inDims[2],inDims[1]),interpolation=cv2.INTER_LINEAR)
			y = cv2.resize(y,(outDims[2],outDims[1]),interpolation=cv2.INTER_LINEAR)

			X[i] = normalize(x)
			Y[i] = normalize(y)

			pathIdx += 1
			if pathIdx > nObs:
				pathIdx = 0 
				shuffle(paths)

		yield X, Y


if __name__ == "__main__": 
	import sys, ipdb, glob

	show = lambda tensor: plt.imshow(hStackBatch(tensor)); plt.show()

	paths = glob.glob("../augmented/x_*")
	feeder = feed(inDims = [4,600,900,3], outDims = [4,10,15,3], paths = paths)
	while True:
		X,Y = feeder.next()
		show(X)



