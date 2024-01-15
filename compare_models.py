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
fig, ax = plt.subplots(3, 2, figsize=(15, 10), squeeze=True,sharey=False)
ax[0][0].axhline(y=evalMSE,color='black',label='Bicubic interpolation')
ax[0][1].axhline(y=np.log(evalMSE),color='black',label='Bicubic interpolation')
ax[1][0].axhline(y=evalMSE,color='black',label='Bicubic interpolation MSE')
ax[1][1].axhline(y=np.log(evalMSE),color='black',label='Bicubic interpolation')
ax[2][0].axhline(y=evalMSE,color='black',label='Bicubic interpolation')
ax[2][1].axhline(y=np.log(evalMSE),color='black',label='Bicubic interpolation')
for i in range(len(savedModels)):
    try:
        savedModel = savedModels[i]
        modelName = savedModel[0]
        path = savedModel[1]
        print(modelName)
        
        train_loss = np.load(path+"\\train_loss.npy")[:100]
        eval_loss = np.load(path+"\\eval_loss.npy")[:100]
        if modelName.find('Res_Physics') != -1:
            ax[2][0].plot(eval_loss, label=modelName)
            ax[2][1].plot(np.log(eval_loss), label=modelName)
        elif modelName.find('Res') != -1:
            ax[1][0].plot(eval_loss, label=modelName)
            ax[1][1].plot(np.log(eval_loss), label=modelName)
        else:
            ax[0][0].plot(eval_loss, label=modelName)
            ax[0][1].plot(np.log(eval_loss), label=modelName)
    except:
        pass
ax[0][0].legend()
ax[1][0].legend()
ax[2][0].legend()
ax[0][0].set_ylabel('MSE')
ax[1][0].set_ylabel('MSE')
ax[2][0].set_ylabel('MSE')


ax[0][1].legend()
ax[1][1].legend()
ax[2][1].legend()
ax[0][1].set_ylabel('log(MSE)')
ax[1][1].set_ylabel('log(MSE)')
ax[2][1].set_ylabel('log(MSE)')

ax[0][0].legend()
ax[1][0].legend()
ax[2][0].legend()
ax[0][1].legend()
ax[1][1].legend()
ax[2][1].legend()

plt.show()
plt.close()