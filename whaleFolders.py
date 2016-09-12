import pandas as pd
import glob as glob
import pdb
import os 

os.chdir("imgs/")
allImgs = glob.glob("w_*")
trCsv = pd.read_csv("../train.csv")

# Make whale directories 
whales = trCsv.whaleID.unique()
for whale in whales:
    if not os.path.exists(whale):
        os.makedirs(whale)

if not os.path.exists("test"):
    os.makedirs("test")
testImages = []

count = 0 
trImgs = []
teImgs = []
for img in allImgs:
    obs = trCsv[trCsv.Image == img]
    if len(obs) > 0: # Then its a training item
        pass
        #os.rename(obs.Image.values[0], obs.whaleID.values[0] + "/" + obs.Image.values[0])
    else:
        os.rename(img,"test/"+img)


