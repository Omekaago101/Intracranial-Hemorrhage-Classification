import pandas as pd
import os
import glob
import numpy as np
 
train = pd.read_csv(os.path.join("D:/Datasets/rsna/", "proc/train/" ))
train_png = glob.glob(os.path.join("D:/Datasets/rsna/", "proc/train/", '*.jpg'))
train_png = [os.path.basename(png)[:-4] for png in train_png]

train_imgs = set(train.Image.tolist())
t_png = [p for p in train_png if p in train_imgs]
t_png = np.array(t_png)
train = train.set_index('Image').loc[t_png].reset_index()