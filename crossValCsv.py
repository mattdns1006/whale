import pandas as pd
import numpy as np
import numpy.random as rng
import sys
import pdb

# Encode the whale names into 1,2,3 ......, 447

# Fn will generate train and test cross validation set from train.csv
def makeCrossValidationCSVs(ratio,subsetSize=10):
    rng.seed(100689)
    df = pd.read_csv("train.csv")
    df = df.sort_values("whaleID")

    whaleNames = df.whaleID.unique()
    nWhales = whaleNames.size

    whaleDict = pd.Series(whaleNames).to_dict()
    whaleDict = {v: k+1 for k, v in whaleDict.items()}
    whaleDictEncode = lambda x: whaleDict.get(x)
    df["label"] = df.whaleID.apply(whaleDictEncode)
    df = df.loc[df["label"]<=subsetSize]
    pdb.set_trace()
    df.reset_index(drop=1,inplace=1)
    
    nObs = df.shape[0]

    rIdx = rng.permutation(nObs)
    df = df.reindex(rIdx)
    df.reset_index(drop=1,inplace=1)

    df.to_csv("trainEncoded.csv")
    cutOff = np.floor(nObs*ratio).astype(np.uint16)
    train,test = df.ix[:cutOff], df.ix[cutOff:]
    train.to_csv("trainCV.csv")
    test.to_csv("testCV.csv")
    print("Train/test shapes = %s/%s" % (train.shape,test.shape))

if __name__ == "__main__":
    try:
        print("Using train/test split ratio of", str(sys.argv[1]))
        ratio = float(sys.argv[1])
    except IndexError:
        print("Split argument not specified, using default ratio of %f." % 0.8)
        ratio = 0.8
    makeCrossValidationCSVs(ratio)
