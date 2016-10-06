import os, glob
from PIL import Image
from tqdm import tqdm
import cv2


def getSize(desiredWidth=600):
    # To keep aspect ratio the same
    imgPaths = glob.glob("../imgs/*/w1_*")
    labelled = glob.glob("../imgs/*/l_*")
    print("Number of labelled images = %d." % len(labelled))
    egImg = Image.open(imgPaths[0])
    ar = (egImg.size[0]/float(egImg.size[1]))
    desiredHeight = int(desiredWidth/ar)

    return desiredWidth, desiredHeight 


def resizeImages(labeledOrOriginal,desiredSize=(500,333),replaceCurrent = False):

    print("Reshaping all images to {0}".format(desiredSize))
    s = str(labeledOrOriginal)
    imgPaths = glob.glob("../imgs/*/"+s+"_*")
    count = 0
    nObs = len(imgPaths)

    for i in tqdm(range(nObs)):
        imgPath = imgPaths[i]
        dst = imgPath.replace(s+"_",s+"S_")
        if os.path.exists(dst) == False or replaceCurrent == True:
            img = cv2.imread(imgPath)
            img = cv2.resize(img,desiredSize,interpolation = cv2.INTER_LINEAR)
            cv2.imwrite(dst,img)
            count += 1
    print ("Added %d more resized images " % count)


if __name__ == "__main__":
    desiredSize = getSize(desiredWidth=900)
    replaceCurrent = True
    resizeImages("w1",desiredSize=desiredSize,replaceCurrent=replaceCurrent) # downscale originals
    resizeImages("l",desiredSize=desiredSize,replaceCurrent=replaceCurrent) # downscale truth labels
