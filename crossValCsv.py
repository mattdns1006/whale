import pandas as pd
import numpy as np
import numpy.random as rng
import sys,os
import ipdb, cv2
from tqdm import tqdm 
def checkSize(df):
        df = df.copy()
	os.chdir("imgs/")
	nObs = df.shape[0]
        indicesToDrop = []
	for i in tqdm(range(nObs)):
		try:
			obs = df.ix[i].fullPath
			if os.path.exists(obs) == False:
                                indicesToDrop.append(i)
				print("{0} not found.".format(obs))
                        else:
				img = cv2.imread(obs)
				
				if img.shape != (600,800,3):
                                        indicesToDrop.append(i)
					print(obs,img.shape)
		except IndexError:
			pass

        count = len(indicesToDrop)
        df.drop(indicesToDrop,inplace=1)
        df.reset_index(inplace=1,drop=1)
	print("Removed {0} files.".format(count))
	return df

# Encode the whale names into 1,2,3 ......, 447
def getFP(row):
	return "../imgs/"+ row.whaleID + "/" + row.Image.replace("w_","head_ss_")

# Fn will generate train and test cross validation set from train.csv
def makeCrossValidationCSVs(ratio):
    rng.seed(100689)
    df = pd.read_csv("train.csv")
    df = df.sort_values("whaleID")

    whaleNames = df.whaleID.unique()
    nWhales = whaleNames.size

    whaleDict = pd.Series(whaleNames).to_dict()
    whaleDict = {v: k for k, v in whaleDict.items()}
    whaleDictEncode = lambda x: whaleDict.get(x)
    df["label"] = df.whaleID.apply(whaleDictEncode)
    df["fullPath"] = df.apply(getFP,1)
    df.drop(["Image","whaleID"],axis=1,inplace=1)
    dfCleaned = checkSize(df)
    def subset():
        df = df.loc[df["label"]<=subsetSize]
        df.reset_index(drop=1,inplace=1)

    while True:
        dfC = dfCleaned.copy()
        nObs = dfC.shape[0]
        rIdx = rng.permutation(nObs)
        dfC = dfC.reindex(rIdx)
        dfC.reset_index(drop=1,inplace=1)

        dfC.to_csv("trainEncoded.csv")

        cutOff = np.floor(nObs*ratio).astype(np.uint16)
        train,test = dfC.ix[:cutOff], dfC.ix[cutOff:]
        break
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
    def weight():
	    gc = train.groupby("label").count()["whaleID"]
	    weights = 1/gc
	    weights.to_csv("trWeights.csv",header=1)
    os.chdir("../")
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
