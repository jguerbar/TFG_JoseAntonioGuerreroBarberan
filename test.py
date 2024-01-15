
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
data_test_y = np.load('.\\data\\testData.npy')

data = data_test_y
root = ".\\savedModels"
savedModels = [ (item, os.path.join(root, item)) for item in os.listdir(root) if not os.path.isfile(os.path.join(root, item)) ]
modelClasses = [item for item, _ in savedModels if item.find('ED') != -1 ]

pd.set_option('display.float_format', '{:.2E}'.format)
df = pd.DataFrame(columns=['ModelName','MSE_Loss','MaxLoss','Physics_Loss'])

#Loss functions
error_loss = MSE_loss()
error_max_loss = MaxLoss()
physics_loss = Navier_Stokes_informed_loss(rho=1, mu=0.00001, dt=1/32,dx=1/255,dy=1/255)
#Interpolators
downsampler = nn.AvgPool2d(UPSAMPLE_FACTOR)
bicubic = nn.Upsample(scale_factor=UPSAMPLE_FACTOR, mode='bicubic')
"""
#Interpolation comparison losses
for data_y in [data]:
    errorLoss_bicubic = 0
    lphysicsInterp = 0
    lphysicsY = 0
    max_error = 0
    for sim_index in range(data_y.shape[0]):
        #Normalize input
        simulationData = data_y[sim_index]
        std = 1
        std = np.std(data_y[sim_index],axis=(0,2,3),keepdims=True)
        simulationData = simulationData/std
        y = torch.tensor(simulationData[:,:,:]).float()
        x = downsampler(y)
        bic_interpolation = bicubic(x)
        errorLoss_bicubic += error_loss(bic_interpolation,y,std).item()
        max_error += error_max_loss(bic_interpolation,y,std).item()

        lphysicsY += physics_loss(y,y,std).item()
        lphysicsInterp += physics_loss(bic_interpolation, y, std).item()
    errorLoss_bicubic /= data_y.shape[0]
    lphysicsInterp /= data_y.shape[0]
    lphysicsY /= data_y.shape[0]
    max_error /= data_y.shape[0]

    print(f'BicubicMSE: {errorLoss_bicubic:.2e}, BicubicMaxMSE: {max_error:.2e},BicubicPhysicsLoss: {lphysicsInterp:.2e}')
    df.loc[len(df)] = ['Bicubic',errorLoss_bicubic,max_error,lphysicsInterp]
"""
for m in range(len(savedModels)):
    savedModel = savedModels[m]
    modelName = savedModel[0]
    isRes = modelName.find('Res') != -1

    path = savedModel[1]
    modelPath = path+"\\"+modelName+".pth"

    model = torch.load(modelPath).to(device)
    model.eval()
    with torch.no_grad():
        avg_error = 0
        avg_max_error = 0
        avg_physics = 0
        avg_bicubic = 0
        avg_bicubic_physics = 0
        n = 0
        for sim_index in range(data.shape[0]):
            simulationData = data[sim_index]
            std = np.std(data[sim_index],axis=(0,2,3),keepdims=True)
            simulationData = simulationData/std
            std = torch.tensor(std).float().to(device)
            for batch_index in range(0,data.shape[1],BATCH_SIZE):
                y = torch.tensor(simulationData[batch_index:batch_index+BATCH_SIZE]).float().to(device)
                x = downsampler(y).to(device)
                output = model(x)
                if isRes: output = bicubic(x)+output
                avg_error += error_loss(output,y,std).item()
                avg_physics += physics_loss(output,y,std).item()
                avg_max_error += error_max_loss(output,y,std).item()
                n += 1
        avg_error /= n
        avg_physics /= n
        avg_max_error /= n
        print(modelName + f' ErrorLoss: {avg_error:.2e}, ErrorRobustLoss: {avg_max_error:.2e}, PhysicsLoss: {avg_physics:.2e}')
        df.loc[len(df)] = [modelName,avg_error,avg_max_error,avg_physics]
df.to_csv('./performance_table.csv')
print(df)    