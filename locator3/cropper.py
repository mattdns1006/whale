import glob, os, cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def rowToCoords(csv,path,imgShape,rescale=0):
    h,w,c = imgShape  
    csvRow = csv[csv["path"] == path]
    x1 = csvRow.x1
    y1 = csvRow.y1
    x2 = csvRow.x2
    y2 = csvRow.y2
    xM = (x2 + x1)/2.0
    yM = (y2 + y1)/2.0
    if rescale == 1:
        x1*=w
        x2*=w
        y1*=h
        y2*=h
        xM*=w
        yM*=h
    return [int(i.values[0]) for i in x1, y1, x2, y2, xM, yM]

if __name__ == "__main__":
    import pdb
    truth = 0
    truthCsv = pd.read_csv("data/0/train.csv")

    modelName = "256_64_0_6_6_4_3.1e-05_20"
    level = 0
    fittedFolder = "fitted/{0}/{1}/test/".format(level,modelName)
    fittedCsv = fittedFolder + "fitted.csv"
    fitted =  pd.read_csv(fittedCsv)
    croppedFolder = fittedFolder + "cropped/"
    flattenList = lambda l: [item for sublist in l for item in sublist]

    df = []
    pad = 400
    error = 0

    if truth == 1:   
        csv = truthCsv
        savePath = "data/1/"
    else:
        csv = fitted
        savePath = croppedFolder
    if not os.path.exists(savePath):
        os.mkdir(savePath)

    csv["path"] = csv.path.apply(lambda x: os.path.abspath(x))
    for f in tqdm(fitted.path[1:]):
        img = cv2.imread(f)
        name = f.split("/")[-1]
        origPath = glob.glob("../imgs/*/{0}".format(name))[0]
        origPath = os.path.abspath(origPath)
        imgO = cv2.imread(origPath)
        h,w,c = imgO.shape  

        x1R, y1R, x2R, y2R, xMR, yMR = rowToCoords(csv,origPath,imgO.shape,rescale=1)  
        imgC = imgO[yMR-pad:yMR+pad,xMR-pad:xMR+pad].copy()
        xCorner, yCorner = xMR-pad, yMR-pad # top left corner of cropped area
        
        x1New, y1New = x1R - xCorner, y1R - yCorner
        x2New, y2New = x2R - xCorner, y2R - yCorner
        if np.any(np.array(imgC.shape)==0):
            error += 1
            continue
        wp = os.path.abspath(savePath + name)
        cv2.imwrite(wp,imgC)
        
        # New (normalized) coords
        h,w,c = imgC.shape
        df.append([wp,float(x1New)/w, float(y1New)/h, float(x2New)/w, float(y2New)/h,w,h])
        
        def show():

            cv2.circle(imgO,(x1R,y1R),30,(125,0,0),-1)
            cv2.circle(imgO,(x2R,y2R),30,(0,0,125),-1)

            cv2.circle(imgC,(x1New,y1New),30,(125,0,0),-1)
            cv2.circle(imgC,(x2New,y2New),30,(0,0,125),-1)

            plt.figure(figsize=(20,5))
            plt.subplot(131)
            plt.imshow(imgO[:,:,::-1]); plt.title(origPath);
            plt.subplot(132)
            plt.imshow(imgC[:,:,::-1]); plt.title(wp);
            plt.subplot(133)
            plt.imshow(img[:,:,::-1]);plt.title(name);
            plt.show()
            
    dfC = pd.DataFrame(df)
    dfC.columns = ["path","x1","y1","x2","y2","w","h"]
    dfC.to_csv(savePath+"train.csv",index=0)

