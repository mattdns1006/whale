import pandas as pd
import numpy as np
import numpy.random as rng
import sys,os,glob
import pdb, cv2, json
from tqdm import tqdm 

def readJson(fp):
    with open(fp) as f:
        data = json.load(f)
    return data

def makePointsCsv():
    df = pd.read_csv("../train.csv")
    imgsPath = "/home/msmith/kaggle/whale/imgs/"

    pfs = ["points1.json","points2.json"]
    dfOut = {}

    points1,points2 = readJson(pfs[0]), readJson(pfs[1])
    i = 0
    for p1 in tqdm(points1):
        assert len(p1["annotations"]) == 1
        fn = p1["filename"]
        for p2_ in points2:
            if p2_["filename"] == fn:
                p2 = p2_ # found corresponding point
                break

        path = df.whaleID[df.Image == p1["filename"]].values
        assert len(path) == 1, "more than one of same filename"
        path = path[0] + "/" + fn
        path = path.replace("w_","w1_")
        path = os.path.join(imgsPath,path)

        if not os.path.exists(path):
            print("{0} does not exist.".format(path))
            continue

        img = cv2.imread(path)
        h,w,c = img.shape

        x1 = p1["annotations"][0]["x"]/w
        y1 = p1["annotations"][0]["y"]/h
        x2 = p2["annotations"][0]["x"]/w
        y2 = p2["annotations"][0]["y"]/h
        dfOut[i] = [path,x1,y1,x2,y2,w,h]
        coords = dfOut[i][1:]
        #eg = cv2.resize(img,(800,800))
        #coords_ = [int(i*800) for i in [x1,y1,x2,y2]]
        i += 1
    dfOut = pd.DataFrame(dfOut).T
    dfOut.columns = ["path","x1","y1","x2","y2","w","h"]
    dfOut.to_csv("train.csv",index=0)
    print("Written path and corresponding points to train.csv")
    print(dfOut.head())

# Encode the whale names into 1,2,3 ......, 447

def getFP(row):
	return "../imgs/"+ row.whaleID + "/" + row.Image.replace("w_","head_")

# Fn will generate train and test cross validation set from train.csv
def makeCrossValidationCSVs(ratio,folder):
    rng.seed(100689)
    df = pd.read_csv(folder+"train.csv")
    nObs = df.shape[0]
    rIdx = rng.permutation(nObs)
    df = df.reindex(rIdx)
    df.reset_index(drop=1,inplace=1)

    cutOff = np.floor(nObs*ratio).astype(np.uint16)
    train,test = df.ix[:cutOff], df.ix[cutOff:]

    train.to_csv(folder+"trainCV.csv",index=0) 
    test.to_csv(folder+"testCV.csv",index=0)
    print("Train/test shapes = %s/%s" % (train.shape,test.shape))

def makeTestCSV(savePath):
    testImgs = glob.glob("../imgs/test/w1_*")
    testImgs.sort()
    print("Len testImgs = {0}".format(len(testImgs)))
    columns = ["path","x1","y1","x2","y2","w","h"]
    df = pd.DataFrame(columns=columns)
    df.path = testImgs
    df.fillna(0,inplace=1)
    df[["x1","y1","x2","y2"]] = df[["x1","y1","x2","y2"]].astype(np.float)
    df.to_csv(savePath,index=0)

if __name__ == "__main__":
    try:
        print("Using train/test split ratio of", str(sys.argv[1]))
        ratio = float(sys.argv[1])
    except IndexError:
        print("Split argument not specified, using default ratio of %f." % 0.8)
        ratio = 0.8
    #makePointsCsv()
    makeTestCSV("data/0/test.csv")
    makeTestCSV("data/1/test.csv")
    makeCrossValidationCSVs(ratio,"data/0/")
    makeCrossValidationCSVs(ratio,"data/1/")
