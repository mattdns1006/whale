import glob
import os
from shutil import copyfile

os.chdir("/home/msmith/kaggle/whale/locator/labels/")
labels = glob.glob("l_*")

for i in range(len(labels)):

    labelPath = labels[i]
    newPath = glob.glob("/home/msmith/kaggle/whale/imgs/*/"+labelPath.replace("l_","w_"))[0].replace("w_","l_")
    print(newPath)
    copyfile(labelPath,newPath)

