import pandas as pd
import numpy as np
import numpy.random as rng
import sys
import ipdb

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
    df.reset_index(drop=1,inplace=1)
    

    while True:
        dfC = df.copy()
        nObs = dfC.shape[0]
        rIdx = rng.permutation(nObs)
        dfC = dfC.reindex(rIdx)
        dfC.reset_index(drop=1,inplace=1)

        dfC.to_csv("trainEncoded.csv")
        cutOff = np.floor(nObs*ratio).astype(np.uint16)
        train,test = dfC.ix[:cutOff], dfC.ix[cutOff:]
        trCount , teCount = [x.groupby("label").count() for x in [train,test]]
        trUnique, teUnique = [x.shape[0] for x in [trCount,teCount]]


        enoughObsTr, enoughObsTe = [any(x<1) for x in [trCount,teCount]]
        enoughUniqueTr, enoughUniqueTe = [x==subsetSize for x in [trUnique,teUnique]]


        bools = [enoughObsTr,enoughObsTe,enoughUniqueTr,enoughUniqueTe]
        okay = all(bools)

        if okay == True:
            print("Class counts for train/test")
            print(trCount)
            print("-"*10)
            print(trUnique)
            print("*"*10)
            print(teCount)
            print("-"*10)
            print(teUnique)
            break
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
