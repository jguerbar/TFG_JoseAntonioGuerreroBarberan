
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss
import numpy as np
import os
import matplotlib.pyplot as plt
#Custom imports
from model import *
from losses import *
import pandas as pd

device = torch.device("cuda")

#Training parameters
UPSAMPLE_FACTOR = 4         #Upsample factor
BATCH_SIZE = 16             #Training Mini-Batch seize
NORMALIZE = True            #Normalization/Standarization of data previous to input.

"""
Load data with shape (N,T,C,W,H) where:
N is the id of the simulation
T is the timestep
C is the channels u,v and p
W is the grid width
H is the grid height
"""
data_test_y = np.load('.\\data\\testData.npy')

data = data_test_y
root = ".\\savedModels"
savedModels = [ (item, os.path.join(root, item)) for item in os.listdir(root) if not os.path.isfile(os.path.join(root, item)) ]
savedModels = [a for a in savedModels if a[0].find('Res') != -1 and a[0].find('Physics') == -1]
#Select data to plot as example. Test from 0-7 First is simulation, second is timestep
INDEXES_PLOT = [(0,20),(1,240),(3,310),(4,310),(7,310)]

#Interpolators
downsampler = nn.AvgPool2d(UPSAMPLE_FACTOR)
bicubic = nn.Upsample(scale_factor=UPSAMPLE_FACTOR, mode='bicubic')
fig, ax = plt.subplots(len(INDEXES_PLOT),len(savedModels)+3,sharex='col',squeeze=True, figsize=(14*2,7*2),sharey='col')
for m in range(len(savedModels)):
    savedModel = savedModels[m]
    modelName = savedModel[0]
    isRes = modelName.find('Res') != -1

    path = savedModel[1]
    modelPath = path+"\\"+modelName+".pth"

    model = torch.load(modelPath).to(device)
    model.eval()
    with torch.no_grad():
        for i in range(len(INDEXES_PLOT)):
            sim, timestep = INDEXES_PLOT[i]

            simulationData = data[sim]
            std = np.std(data[sim],axis=(0,2,3),keepdims=True)
            simulationData = simulationData/std
            std = torch.tensor(std).float().to(device)

            y = torch.tensor(simulationData[[timestep]]).float().to(device)
            x = downsampler(y).to(device)
            original_lr = x.cpu().numpy()
            bicubic_reconstruction = bicubic(x).cpu().numpy()
            output_model = model(x)
            if isRes: output_model = bicubic(x)+output_model
            model_reconstruction = output_model.cpu().numpy()
            original_hr = y.cpu().numpy()
            std = std.cpu().numpy()
            
            #Denormalize
            original_lr = original_lr*std
            bicubic_reconstruction = bicubic_reconstruction*std
            model_reconstruction = model_reconstruction*std
            original_hr = original_hr*std

            original_lr = original_lr[0]
            bicubic_reconstruction = bicubic_reconstruction[0]
            model_reconstruction = model_reconstruction[0]
            original_hr = original_hr[0]



            original_lr = original_lr[2]#np.sqrt(original_lr[0]**2+original_lr[1]**2) #original_lr[2]#
            original_lr = original_lr.repeat(4, axis = 0).repeat(4, axis = 1) #Upsample by repeating pixels so they are represented in the same scale
            bicubic_reconstruction = bicubic_reconstruction[2]#np.sqrt(bicubic_reconstruction[0]**2+bicubic_reconstruction[1]**2)
            model_reconstruction = model_reconstruction[2]#np.sqrt(model_reconstruction[0]**2+model_reconstruction[1]**2)
            original_hr = original_hr[2]#np.sqrt(original_hr[0]**2+original_hr[1]**2)
                
            #Represent a small square of of 80 pixels, centered in the middle
            size = 80
            x_min = 128-round(size/2)
            x_max = x_min+size
            y_min = 128-round(size/2)
            y_max = y_min+size

            original_lr = original_lr[x_min:x_max,y_min:y_max]
            bicubic_reconstruction = bicubic_reconstruction[x_min:x_max,y_min:y_max]
            model_reconstruction = model_reconstruction[x_min:x_max,y_min:y_max]
            original_hr = original_hr[x_min:x_max,y_min:y_max]
                
            ax[i][m+2].imshow(model_reconstruction, cmap='jet', vmin=model_reconstruction.min(), vmax=model_reconstruction.max(), origin='lower')
            if i==0: ax[i][m+2].set_title(modelName)
            ax[i][m+2].set_axis_off()

            if m == 0: #For first model, also plot the original, bicubic and 
                ax[i][0].imshow(original_lr, cmap='jet', vmin=original_lr.min(), vmax=original_lr.max(), origin='lower')
                if i==0:ax[i][0].set_title('LowRes')
                ax[i][0].set_axis_off()
                ax[i][1].imshow(bicubic_reconstruction, cmap='jet', vmin=bicubic_reconstruction.min(), vmax=bicubic_reconstruction.max(), origin='lower')
                if i==0:ax[i][1].set_title('Bicubic')
                ax[i][1].set_axis_off()
                ax[i][-1].imshow(original_hr, cmap='jet', vmin=original_hr.min(), vmax=original_hr.max(), origin='lower')
                if i==0:ax[i][-1].set_title('Ground-Truth')
                ax[i][-1].set_axis_off()

plt.savefig("./test.png")
plt.show()
plt.close()
exit()
    