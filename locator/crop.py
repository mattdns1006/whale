import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pylab import rcParams
from scipy.ndimage.interpolation import rotate, zoom
import scipy as sp
import cv2
get_ipython().magic('matplotlib inline')
import sys
from tqdm import tqdm
import argparse

sys.path.append("/home/msmith/misc/pcaRotate/") # Helper function
from pcaRotate import main as rotate
from pcaRotate import getRed
from pylab import rcParams
rcParams["figure.figsize"] = 30,25


# In[ ]:

if __name__ == "__main__": 
    imgPaths = glob.glob("/home/msmith/kaggle/whale/imgs/whale*/m1_ss_*")
    #testPaths = glob.glob("/home/msmith/kaggle/whale/imgs/test*/m1*")
    print("Make sure image is in RGB order not BGR")
    imgPaths.sort()
    maskPath = "m1_ss_"
    dstPath = "head_ss_"
    
    aspectRatio = 1.5
    h = 400
    w = int(h*aspectRatio)
    
    for path in tqdm(np.random.permutation(imgPaths)):
        
        orig, mask = [cv2.imread(x)[:,:,::-1] for x in [path.replace(maskPath,"w1_"),path]]
        croppedHead, origRot, maskRot, red = rotate(orig=orig,mask=mask,ellipseThresh=10,redThresh=[0.1,0.2],cntThresh=0.01,pad=20,aspectRatio=aspectRatio)
        croppedHead = cv2.resize(croppedHead,(w,h),interpolation=cv2.INTER_LINEAR)
        wp = path.replace(maskPath,dstPath)
        cv2.imwrite(wp,croppedHead)
        
        if np.random.uniform() < 0.01:
            plt.subplot(161)
            plt.imshow(orig)
            plt.subplot(162)
            plt.imshow(mask)
            plt.subplot(163)
            plt.imshow(maskRot,cmap=cm.gray)
            plt.subplot(164)
            plt.imshow(red,cmap=cm.gray)
            plt.subplot(165)
            plt.imshow(origRot)
            plt.subplot(166)
            plt.imshow(croppedHead)
            plt.show()

