import numpy as np
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torchmetrics
import matplotlib
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import interpolate

#Custom imports
from model import *
from losses import *
import pandas as pd

root = ".\\savedModels"
savedModels = [ (item, os.path.join(root, item)) for item in os.listdir(root) if not os.path.isfile(os.path.join(root, item)) ]

evalMSE = 2.11e-03

fig, ax = plt.subplots(len(modelClasses), 2, figsize=(15, 10), squeeze=True,sharex=True)

for j in range(len(modelClasses)):
    modelClassName = modelClasses[j]
    print(modelClassName)
    ax[j][0].axhline(y=evalMSE,color='black',label='Bicubic interpolation')
    ax[j][1].axhline(y=np.log(evalMSE),color='black',label='Bicubic interpolation')
    ax[j][0].set_title(modelClassName)
    ax[j][1].set_title(modelClassName + ' log losses')

    for i in range(len(savedModels)):
        savedModel = savedModels[i]
        modelName = savedModel[0]
        path = savedModel[1]

        if modelName[:len(modelClassName)] != modelClassName: continue
        train_loss = np.load(path+"\\train_loss.npy")[:100]
        eval_loss = np.load(path+"\\eval_loss.npy")[:100]

        ax[j][0].plot(eval_loss, label=modelName)
        ax[j][1].plot(np.log(eval_loss), label=modelName)
        ax[j][0].set_ylabel('MSE')
        if modelName.find('Res')==-1:ax[j][0].set_ylim(top=max(evalMSE*1.3,eval_loss[40]),bottom=0)

        ax[j][1].set_ylabel('log(MSE)')
    ax[j][0].legend()
    ax[j][1].legend()

plt.show()
plt.close()