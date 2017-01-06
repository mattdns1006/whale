import pandas as pd
import numpy as np
import numpy.random as rng
import sys,os
import ipdb, cv2
from tqdm import tqdm 

# Encode the whale names into 1,2,3 ......, 447

def getFP(row):
	return "../imgs/"+ row.whaleID + "/" + row.Image.replace("w_","head_")

def saveEncoding(whaleDict):
    import pickle
    '''
    Saves whale ids mapping to encoded label for submission file
    '''
    print("Saving whale encoding")
    pickle.dump(whaleDict,open("encoding.p","wb"))

# Fn will generate train and test cross validation set from train.csv
def makeCrossValidationCSVs(ratio):
    rng.seed(100689)
    df = pd.read_csv("../train.csv")
    df = df.sort_values("whaleID")

    whaleNames = df.whaleID.unique()
    nWhales = whaleNames.size

    whaleDict = pd.Series(whaleNames).to_dict()
    whaleDict = {v: k for k, v in whaleDict.items()}
    whaleDictEncode = lambda x: whaleDict.get(x)
    saveEncoding(whaleDict)
    df["label"] = df.whaleID.apply(whaleDictEncode)
    df["fullPath"] = df.apply(getFP,1)
    df.drop(["Image","whaleID"],axis=1,inplace=1)

    def subset():
        df = df.loc[df["label"]<=subsetSize]
        df.reset_index(drop=1,inplace=1)

    nObs = df.shape[0]

    exists = lambda row: os.path.exists(row.fullPath)
    ipdb.set_trace()
    df = df.drop(df[df.apply(exists,1)==False].index) # remove paths if do not exists
    df.reset_index(drop=1,inplace=1)
    nObsAfter = df.shape[0]
    rIdx = rng.permutation(nObsAfter)
    df = df.reindex(rIdx)

    print("Lost {0} observations in cropping".format(nObs-nObsAfter))
    df.reset_index(drop=1,inplace=1)

    cutOff = np.floor(nObs*ratio).astype(np.uint16)
    train,test = df.ix[:cutOff], df.ix[cutOff:]

    train.to_csv("trainCV.csv",index=0)
    test.to_csv("testCV.csv",index=0)
    print("Train/test shapes = %s/%s" % (train.shape,test.shape))

if __name__ == "__main__":
    try:
        print("Using train/test split ratio of", str(sys.argv[1]))
        ratio = float(sys.argv[1])
    except IndexError:
        print("Split argument not specified, using default ratio of %f." % 0.8)
        ratio = 0.8
    makeCrossValidationCSVs(ratio)
