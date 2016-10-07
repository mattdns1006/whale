import os,sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import cv2
import tensorflow as tf
sys.path.insert(0,"/home/msmith/misc/py/")
import aug # Augmentation

def oneHotEncode(label,nClasses):
    y = np.zeros(nClasses)
    y[label - 1] = 1
    return y

def oneHotDecode(ohVector):
    return ohVector.argmax() + 1

class dataGenerator():
    def __init__(self,trainOrTest,bS,inputSize=(300,300,3)):
        assert trainOrTest in ("train","test"), "trainOrTest argument must be 'train' or 'test'"
        if trainOrTest == "train":
            self.csv = pd.read_csv("../trainCV.csv")
            self.aug = 1
        else:
            self.csv = pd.read_csv("../testCV.csv")
            self.aug = 0

        self.nObs = self.csv.shape[0]
        self.bS = bS #batchSize
        assert len(inputSize) == 3, "Image must be dim 3"
        self.inputSize = inputSize
        self.w,self.h,self.c = self.inputSize
        self.tensorShape = (self.bS,self.h,self.w,self.c)
        self.nClasses = self.csv.label.max()
        self.whaleLookUp = self.csv[["whaleID","label"]].drop_duplicates().sort_values("label").reset_index(drop = 1)

    def getPath(self,row):
        return "../imgs/"+ row.whaleID + "/" + row.Image.replace("w_","head_")

    def shuffle(self):
        rIdx = np.random.permutation(self.nObs)
        self.csv = self.csv.reindex(rIdx)
        self.csv.reset_index(drop=1,inplace=1)

    def decodeToName(self,ohVector):
	names = list()
	for i in range(ohVector.shape[0]):
		label = oneHotDecode(ohVector[i])
		loc = self.whaleLookUp[self.whaleLookUp.label==label]
		name = loc.whaleID.values[0][6:] + " ("+str(label)+") "
		names.append(name)
        return names

    def generator(self):
        self.idx = 0
        while True:
            X = np.empty(self.tensorShape).astype(np.float32)
            Y = np.empty((self.bS,self.nClasses)).astype(np.float32)
            for i in range(self.bS):
                obs = self.csv.loc[self.idx]
                path = self.getPath(obs)
                self.idx +=1
                x = cv2.imread(path)
                x = cv2.resize(x,(self.w,self.h),interpolation= cv2.INTER_LINEAR)
                if self.aug == 1:
                    x = aug.gamma(x,0.01)
                    x = aug.rotateScale(x,maxAngle=6,maxScaleStd=0.03)
                    
                x = (x - x.mean())/x.std()
                X[i] = x
                Y[i] = oneHotEncode(obs.label,self.nClasses)
                if self.idx == self.nObs:
                    self.idx = 0
                    self.shuffle()
            yield X,Y


if __name__ == "__main__":
    eg = dataGenerator("train",bS=4,inputSize=(250,250,3))
    gen = eg.generator()
    for i in tqdm(range(2)):
        X,Y = next(gen)
