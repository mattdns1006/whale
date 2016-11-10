import os
import glob
import numpy as np
import cv2
from pylab import rcParams
from scipy.ndimage.interpolation import rotate, zoom, shift
import time
from tqdm import tqdm
import sys, ipdb
sys.path.append("/home/msmith/misc/py")
from removeFiles import removeFiles
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

def aug():
    newWidth, newHeight = 900,600
    labels = glob.glob("../imgs/*/lS_*")
    nImgs = len(labels)
    nAug = 30 
    print("{0} labels to be augmented {1} times to size {2}.".format(nImgs,nAug,(newWidth,newHeight)))
    nTrain = int(0.8*nImgs)*nAug
    nTest = int(0.2*nImgs)*nAug
    print("Number of unique labels = train/test = {0}/{1}".format(nTrain/nAug,nTest/nAug))
    print("First {0} will be training, rest {1} testing".format(nTrain,nTest))

    if not os.path.exists("augmented/train"):
        os.makedirs("augmented/train")
        os.makedirs("augmented/testAugmented")
        os.makedirs("augmented/test")

    imgNo = 0

    for i in tqdm(range(nImgs)[:],"Images to augment"):
        y = labels[i]
        x = y.replace("lS_","w1S_")
        imgX, r = cv2.imread(x), cv2.imread(y)

        notRedYellow = (r[:,:,2] < 200) & (r[:,:,1] < 200) | (r[:,:,0] > 100)
        r[notRedYellow] = 0

        
        for i in tqdm(range(nAug)[:],desc=""):

            imgX_, r_ = imgX.copy(), r.copy()
            if i > 0:

                start = time.time()
                angle = np.random.uniform(-20,20)
                scale = np.random.normal(1.0,0.1)


                # Augment Y (YUV) here

                #mapping = (cdfAug(mvn.rvs(1),np.linspace(0,1,256))*255).astype(np.uint32).clip(0,255)
                #imgX_ = histMatch.fitMapping(imgX_,mapping)

                shiftX, shiftY, _ = np.random.normal(0,30,3)

                M = cv2.getRotationMatrix2D((newWidth/2,newHeight/2),angle,scale=scale)
                M[0,1] += np.random.normal(0,0.2)
                
                M[0,2] = shiftX
                M[1,2] = shiftY
                if np.random.uniform() < 0.5:
                    imgX_,r_ = [cv2.flip(img,1) for img in [imgX_,r_]]

                if np.random.uniform() < 0.5:
                    imgX_,r_ = [cv2.flip(img,0) for img in [imgX_,r_]]

                imgX_, r_ = [cv2.warpAffine(img,M,(newWidth,newHeight),borderMode = 0,flags=cv2.INTER_CUBIC) for img in [imgX_,r_]]

            if imgNo >= nTrain:
                if i == 0:
                    path = "augmented/test/"

                else:
                    path = "augmented/testAugmented/"

            else:
                path = "augmented/train/"

            cv2.imwrite(path+"x_"+str(imgNo)+".jpg",imgX_)
            cv2.imwrite(path+"y_"+str(imgNo)+".jpg",r_)
            
            imgNo += 1

if __name__ == "__main__":
    while True:
        delete = raw_input("Delete current augmented images? (y/n)") 
        if delete in ("y","n"):
            break
    if delete == "y":
        removeFiles("augmented/testAugmented/",check=0)
        removeFiles("augmented/train/",check=0)
        removeFiles("augmented/test/",check=0)

    aug()

  


