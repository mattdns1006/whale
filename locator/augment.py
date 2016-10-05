import os
import glob
import numpy as np
from PIL import Image
from pylab import rcParams
from scipy.ndimage.interpolation import rotate, zoom, shift
import time
from tqdm import tqdm

if __name__ == "__main__":
    newWidth, newHeight = 500,333
    labels = glob.glob("../imgs/*/lS_*")
    nImgs = len(labels)
    nAug = 15
    print("{0} labels to be augmented {1} times to size {2}.".format(nImgs,nAug,(newWidth,newHeight)))

    imgNo = 0
    for i in tqdm(range(nImgs)[:],"Images to augment"):
        y = labels[i]
        x = y.replace("lS_","w1S_")
        imgX, imgY = Image.open(x), Image.open(y)
        r = np.array(imgY)
        notRedYellow = (r[:,:,0] < 200) & (r[:,:,1] < 200) | (r[:,:,2] > 100)
        r[notRedYellow] = 0
        redYellow = np.invert(notRedYellow)
        r = Image.fromarray(r)
        
        for i in tqdm(range(nAug),desc=""):
            
            start = time.time()
            angle = np.random.uniform(-30,30)
            imgX_, r_ = imgX.copy(), r.copy()
            shiftX, shiftY, _ = np.random.normal(0,50,3)
            
            
            if np.random.uniform() < 0.5:
                imgX_, r_ = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in [imgX_,r_]]
            if np.random.uniform() < 0.3:
                imgX_, r_ = [img.transpose(Image.FLIP_TOP_BOTTOM) for img in [imgX_,r_]]
            imgX_ = rotate(imgX_,angle,mode="constant")
            r_ = rotate(r_,angle,mode="constant")
            
            imgX_ = shift(imgX_,(shiftX, shiftY,0),mode="constant")
            r_ = shift(r_,(shiftX, shiftY,0),mode="constant")
            
            imgX_, r_ = [Image.fromarray(img) for img in [imgX_,r_]]
            imgX_, r_ = [img.resize((newWidth,newHeight)) for img in [imgX_,r_]]
            
            imgX_.save("augmented/x_"+str(imgNo)+".jpg")
            r_.save("augmented/y_"+str(imgNo)+".jpg")
            
            imgNo += 1

