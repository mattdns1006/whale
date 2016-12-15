import pandas as pd
import glob
import numpy as np

if __name__ == "__main__":
    import ipdb

    for x in ["trainAug","testAug","test"]:
        paths = glob.glob("data/{0}/*/*".format(x))

        df = pd.DataFrame({"path":paths})
        lab = lambda path: path.split('/')[2]
        df["label"] = df.path.apply(lab)
        df.to_csv("{0}.csv".format(x))
    
