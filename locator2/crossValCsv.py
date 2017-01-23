import pandas as pd
import numpy as np
import numpy.random as rng
import sys,os,glob
import pdb, cv2, json
from tqdm import tqdm 

def makeCsv():
    if not os.path.exists("csvs/"):
        os.mkdir("csvs")
    for d in ["train","test","testAugmented"]:
        x = glob.glob("../locator/augmented/{0}/x_*.jpg".format(d))[:20]
        x = [os.path.abspath(i) for i in x]
        y = [i.replace("x_","y_") for i in x]
        df = pd.DataFrame({"x":x,"y":y})
        wp = "csvs/{0}.csv".format(d)
        df.to_csv(wp,index=0)
        print("{0} created with {1} observations.".format(wp,df.shape))

if __name__ == "__main__":
    makeCsv()
