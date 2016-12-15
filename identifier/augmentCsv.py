import pandas as pd
import glob
import numpy as np

if __name__ == "__main__":
    import ipdb

    for x in ["allAug","trainAug","testAug","test"]:
        paths = glob.glob("data/{0}/*/*".format(x))

        df = pd.DataFrame({"fullPath":paths})
        lab = lambda path: path.split('/')[2]
        df["label"] = df.fullPath.apply(lab)
        df = df[["label","fullPath"]] # swap order
        df.to_csv("{0}.csv".format(x),index=0)
    
