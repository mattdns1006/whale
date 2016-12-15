import os, glob, sys, time, cv2
import numpy as np
import pandas as pd
from pylab import rcParams
from scipy.ndimage.interpolation import rotate, zoom, shift
from tqdm import tqdm
import shutil

sys.path.append("/home/msmith/misc/py")
from removeFiles import removeFiles
import matplotlib.pyplot as plt

def show(img):
    plt.imshow(img)
    plt.show()

def makeDir(path):
    if not os.path.exists(path):
        os.makedirs(path) 

def makedirs():
    os.makedirs("data/trainAug") # all augmented tr
    os.makedirs("data/testAug") # all augmented te - hence add tr to te to get total
    os.makedirs("data/test") # single test image 

def aug(trOrTe, nAug=10, outShape=(350,350)):
    oW, oH = outShape
    if trOrTe == "train":
        csv = pd.read_csv("../trainCV.csv")
        savePath = "data/trainAug/"
    elif trOrTe == "test":
        csv = pd.read_csv("../testCV.csv")
        savePath = "data/testAug/"
    nObs = csv.shape[0]
    csv.sort_values("label",inplace=1)

    for i in tqdm(range(nObs)):

        obs = csv.iloc[i]
        path = obs.fullPath
        label = obs.label
        if os.path.exists == False:
            continue
        makeDir(path)
        img = cv2.imread(path)
        w, h, c = img.shape

        for j in range(nAug):
            imgC = img.copy()
            wp = savePath + str(label) + "/"
            makeDir(wp)
            if j > 0:
                angle = np.random.uniform(-10,10)
                scale = np.random.normal(1.0,0.1)
                shiftX = np.random.normal(0,w*0.01,1)
                shiftY = np.random.normal(0,h*0.01,1)
                M = cv2.getRotationMatrix2D((w/2,h/2),angle,scale=scale)
                M[0,1] += np.random.normal(0,0.2)
                M[0,2] = shiftX
                M[1,2] = shiftY
                imgC = cv2.warpAffine(img,M,(w,h),borderMode = 0,flags=cv2.INTER_CUBIC)
            elif j == 0 and trOrTe == "test":
                wp = "data/test/" + str(label) + "/"
                makeDir(wp)
            number = len(glob.glob(wp+"*"))
            wp += "{0}.jpg".format(number)
            imgC = cv2.resize(imgC,outShape,interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(wp,imgC)

if __name__ == "__main__":

    import ipdb
    try:
        [shutil.rmtree(x) for x in ["data/trainAug","data/testAug","data/test"]]
    except OSError:
        "Doesn't exist"
    makedirs()
    for x in ["train","test"]:
        aug(x)

