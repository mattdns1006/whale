import pandas as pd
import glob as glob
import pdb
import os 

os.chdir("imgs/")
allImgs = glob.glob("w_*")
trCsv = pd.read_csv("../train.csv")

# Make whale directories 
def makeDirs():
    whales = trCsv.whaleID.unique()
    for whale in whales:
        if not os.path.exists(whale):
            os.makedirs(whale)

    if not os.path.exists("test"):
        os.makedirs("test")


def allocateToDirs(): # allocates the whale into its designated whale folder
    testImages = []
    trImgs = []
    teImgs = []
    for img in allImgs:
        obs = trCsv[trCsv.Image == img]
        if len(obs) > 0: # Then its a training item as its not in train csv.
            os.rename(obs.Image.values[0], obs.whaleID.values[0] + "/" + obs.Image.values[0])
        else:
            pass
            os.rename(img,"test/"+img)


