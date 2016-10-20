import os
import glob
import numpy as np
import cv2
from pylab import rcParams
from scipy.ndimage.interpolation import rotate, zoom, shift
import time
from tqdm import tqdm
import sys, pdb
sys.path.append("/home/msmith/misc/py")
sys.path.append("/home/msmith/kaggle/whale/colorPerturbation")
from removeFiles import removeFiles
import matplotlib.pyplot as plt
import ipdb
import pandas as pd
from scipy import stats
import histMatch

def aug():
    newWidth, newHeight = 900,600
    labels = glob.glob("../imgs/*/lS_*")
    nImgs = len(labels)
    nAug = 100 
    print("{0} labels to be augmented {1} times to size {2}.".format(nImgs,nAug,(newWidth,newHeight)))

    imgNo = 0
    wCdfCsv = pd.read_csv("../colorPerturbation/weights/weights.csv") # Weights for augmentation of YUV
    mu = np.array(wCdfCsv.mean())
    mvn = stats.multivariate_normal(mu,0.001)

    def cdfAug(w,x):
        w1, w2, w3, w4 = w
        return w1*np.tanh(((x-w2)/w3)) + w4

    for i in tqdm(range(nImgs)[:],"Images to augment"):
        y = labels[i]
        x = y.replace("lS_","w1S_")
        imgX, r = cv2.imread(x), cv2.imread(y)

        notRedYellow = (r[:,:,2] < 200) & (r[:,:,1] < 200) | (r[:,:,0] > 100)
        r[notRedYellow] = 0
        
        for i in tqdm(range(nAug),desc=""):

            
            start = time.time()
            angle = np.random.uniform(-20,20)
            scale = np.random.normal(1.0,0.1)
            imgX_, r_ = imgX.copy(), r.copy()

            # Augment Y (YUV) here

            mapping = (cdfAug(mvn.rvs(1),np.linspace(0,1,256))*255).astype(np.uint32).clip(0,255)
            imgX_ = histMatch.fitMapping(imgX_,mapping)

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

            cv2.imwrite("augmented/x_"+str(imgNo)+".jpg",imgX_)
            cv2.imwrite("augmented/y_"+str(imgNo)+".jpg",r_)
            
            imgNo += 1

if __name__ == "__main__":
    while True:
        delete = raw_input("Delete current augmented images? (y/n)") 
        if delete in ("y","n"):
            break
    if delete == "y":
        #removeFiles("augmented/",check=1)
        removeFiles("augmented/",check=0)

    aug()

  


