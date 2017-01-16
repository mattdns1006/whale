import pandas as pd
import numpy as np
import numpy.random as rng
import sys,os
import ipdb, cv2
from tqdm import tqdm 

# Encode the whale names into 1,2,3 ......, 447

def getFP(row):
	return "../imgs/"+ row.whaleID + "/" + row.Image.replace("w_","head_")


# Fn will generate train and test cross validation set from train.csv
def makeCrossValidationCSVs(ratio,dir):
    rng.seed(100689)
    df = pd.read_csv(dir+"train.csv")
    nObs = df.shape[0]
    rIdx = rng.permutation(nObs)
    df = df.reindex(rIdx)

    cutOff = np.floor(nObs*ratio).astype(np.uint16)
    train,test = df.ix[:cutOff], df.ix[cutOff:]

    train.to_csv(dir+"trainCV.csv",index=0) 
    test.to_csv(dir+"testCV.csv",index=0)
    print("Train/test shapes = %s/%s" % (train.shape,test.shape))

if __name__ == "__main__":
    try:
        print("Using train/test split ratio of", str(sys.argv[1]))
        ratio = float(sys.argv[1])
    except IndexError:
        print("Split argument not specified, using default ratio of %f." % 0.8)
        ratio = 0.8
    makeCrossValidationCSVs(ratio,dir="")
    makeCrossValidationCSVs(ratio,dir="cropped/truth/")
