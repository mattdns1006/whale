import cv2, sys, os
import pandas as pd
import numpy as np
from random import shuffle
sys.path.append("/home/msmith/misc/histMatch/")
from histMatch import histMatchAllChannels
import matplotlib.pyplot as plt

def normalize(img):
	img = img.astype(np.uint8)
	img = img/255.0
	return img

def feedVideo(inDims,filePath,matchingImg="x_1"):

        baseMatchImage = cv2.imread("/home/msmith/kaggle/whale/locator/augmented/histMatchBase/{0}.jpg".format(matchingImg))
        cap = cv2.VideoCapture(filePath)
        while not cap.isOpened():
            cap = cv2.VideoCapture(filePath)
            cv2.waitKey(1000)

        posFrame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = cap.read()
            x = cv2.resize(frame,(inDims[2],inDims[1]),interpolation=cv2.INTER_LINEAR)
            orig = x.copy()
            x, _ = histMatchAllChannels(x,baseMatchImage)
            x = normalize(x)
	    x = np.expand_dims(x,0) 
            yield x,orig

def feed(inDims, outDims, paths, matchingImg = "x_1", histMatch = 1):
	
        paths = paths
	nObs = len(paths)
	X = np.zeros(inDims).astype(np.float32)
	Y = np.zeros(outDims).astype(np.float32)
	batchSize = inDims[0]
	pathIdx = 0
	shuffle(paths)
        finished = 0
        baseMatchImage = cv2.imread("../augmented/histMatchBase/{0}.jpg".format(matchingImg))

	while True:

                imgPaths = []
		for i in range(batchSize):
			path = paths[pathIdx]
                        imgPaths.append(path)
			x = cv2.imread(path)
			x = cv2.resize(x,(inDims[2],inDims[1]),interpolation=cv2.INTER_LINEAR)
                        if histMatch == 1:
                            x, _ = histMatchAllChannels(x,baseMatchImage)

                        if os.path.exists(path.replace("x_","y_")):
			    y = cv2.imread(path.replace("x_","y_"))
			    y = cv2.resize(y,(outDims[2],outDims[1]),interpolation=cv2.INTER_LINEAR)
			    Y[i] = normalize(y)
                        else:
                            y = np.zeros((outDims[2],outDims[1],outDims[0])).fill(255)

			X[i] = normalize(x)
			Y[i] = normalize(y)


			pathIdx += 1
			if pathIdx >= nObs:
				pathIdx = 0 
                                finished = 1

                        if pathIdx % 1000 == 0:
                            print("{0} of {1}".format(pathIdx,nObs))
                                    
                yield X, Y, finished, imgPaths


if __name__ == "__main__": 
	import sys, ipdb, glob
        import matplotlib.pyplot as plt


	paths = glob.glob("../augmented/test/x_*")
	feeder = feed(inDims = [4,600,900,3], outDims = [4,10,15,3], paths = paths)
	for i in xrange(300):
		X,Y,fin = feeder.next()



