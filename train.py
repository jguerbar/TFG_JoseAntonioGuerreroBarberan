
import numpy as np
import math
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
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
import random

#Hyperparameters
device = torch.device("cuda")
torch.cuda.empty_cache()
N_EPOCHS = 30
TRAIN_FRACTION = 1/2 #Which fraction of data will be used for training
EVAL_FRACTION = 1/2  #Which fraction of non train data will be used for evaluation
DOWNSAMPLE_FACTOR = 4
BATCH_SIZE = round(320/20)
INPUT_CHANNELS = 3
PREDICT_RESIDUAL = False
OPTIMIZE_PHYSICS = False
PROB_PHYSICS = 0.1
SEQ_SIZE = None #8
NORMALIZE = True

#Import data
data_hr = np.load('.\\data\\allData.npy')

"""
Transpose axis, giving shape (N,T,C,W,H) where:
N is the id of the simulation
T is the timestep
C is the channels u,v and p
W is the grid width
H is the grid height
"""
data_hr = np.transpose(data_hr,(0,1,4,2,3))

#Separate data
indexTrain = round(TRAIN_FRACTION*data_hr.shape[0])
indexEval = round((TRAIN_FRACTION+((1-TRAIN_FRACTION)*EVAL_FRACTION))*data_hr.shape[0])+1

data_train_y = data_hr[:indexTrain]
data_eval_y = data_hr[indexTrain:indexEval]
data_test_y = data_hr[indexEval:]

print(data_train_y.shape)
print(data_eval_y.shape)
print(data_test_y.shape)



#Create model instance
#model = NaiveTransConv(INPUT_CHANNELS, 256).to(device)
#model = ED_UpsamplingModel(INPUT_CHANNELS).to(device)
model = FSRCNN(INPUT_CHANNELS,d=56,s=12,m=10,scale_factor=DOWNSAMPLE_FACTOR).to(device)
#model = SRCNN(INPUT_CHANNELS,64,32).to(device)
#model = VDSR(numInputChannels=INPUT_CHANNELS,deep_channels=64,num_layers=20,scale_factor=4).to(device)
#model = ESPCN(INPUT_CHANNELS,INPUT_CHANNELS,64,DOWNSAMPLE_FACTOR).to(device) 

#Optimizer
weight_decay = 1e-7
learning_rate = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

#Loss functions
error_loss = PolymorphicMSE_loss()
physics_loss = Navier_Stokes_informed_loss(rho=1, mu=0.00001, dt=1/32,dx=1/255,dy=1/255)

#Interpolation comparison losses
bicubic = nn.Upsample(scale_factor=DOWNSAMPLE_FACTOR, mode='bicubic')
downsampler = nn.AvgPool2d(DOWNSAMPLE_FACTOR)
for data_y in [data_train_y,data_eval_y,data_test_y]:
    errorLoss_bicubic = 0
    lphysicsInterp = 0
    lphysicsY = 0
    for sim_index in range(data_y.shape[0]):
        #Normalize input
        simulationData = data_y[sim_index]
        std = 1
        if NORMALIZE:
            std = np.std(data_y[sim_index],axis=(0,2,3),keepdims=True)
            simulationData = simulationData/std

        y = torch.tensor(simulationData[:,:,:]).float()
        x = downsampler(y)
        bic_interpolation = bicubic(x)
        errorLoss_bicubic += error_loss(bic_interpolation,y,std).item()
        lphysicsInterp += physics_loss(bic_interpolation, bic_interpolation, std)
        lphysicsY += physics_loss(y,y,std)
    errorLoss_bicubic /= data_y.shape[0]
    lphysicsInterp /= data_y.shape[0]
    lphysicsY /= data_y.shape[0]

    print(f'BicubicMSE: {errorLoss_bicubic:.2e}, PhysicsLossInterp: {lphysicsInterp:.2e}, PhysicsLossY: {lphysicsY:.2e}')

def epoch_trainer(data,loss_function,loss_function_aux=None,eval=True):
    downsampler = nn.AvgPool2d(DOWNSAMPLE_FACTOR)
    avg_loss = 0
    n = 0

    if eval: model.eval()
    else: model.train()

    for sim_index in range(data.shape[0]):
        simulationData = data[sim_index]
        std = 1
        if NORMALIZE:
            std = np.std(data[sim_index],axis=(0,2,3),keepdims=True)
            simulationData = simulationData/std
            std = torch.tensor(std).float().to(device)

        sim_downsampled = downsampler(torch.tensor(simulationData[:,:,:]))
        sim_reupsampled = bicubic(sim_downsampled)
        sim_downsampled = sim_downsampled.numpy()
        sim_reupsampled = sim_reupsampled.numpy()

        for batch_index in range(0,data.shape[1],BATCH_SIZE):
            if SEQ_SIZE != None:
                x = []
                y = []
                for i in range(BATCH_SIZE-SEQ_SIZE):
                    y.append(simulationData[batch_index+i:batch_index+SEQ_SIZE+i,:,:])
                    x.append(sim_reupsampled[batch_index+i:batch_index+SEQ_SIZE+i,:,:])
                x = torch.tensor(np.array(x)).float()
                y = torch.tensor(np.array(y)).float()
            else:
                y = torch.tensor(simulationData[batch_index:batch_index+BATCH_SIZE,:,:,:]).float()
                x = torch.tensor(downsampler(y))
            #Move to device, for efficient use of VRAM            
            x = x.to(device)
            y = y.to(device)
            #if not eval, randomly rotate matrix on spatial dimentions
            if not eval:
                k = np.random.randint(low=0,high=4)
                x = torch.rot90(x,k,dims=(-1,-2))
                y = torch.rot90(y,k,dims=(-1,-2))

            output = model(x) # forwards pass
            if SEQ_SIZE != None: #Take only last sequence
                x = x[:,-1,:,:,:]
                y = y[:,-1,:,:,:]

            if PREDICT_RESIDUAL: #Reconstruct when modelling for PREDICT_RESIDUAL
                #if SEQ_SIZE != None: output = x + output
                #else: output = bicubic(x)+output
                output = bicubic(x)+output
            loss_train = loss_function(output,y,std)
            if (np.isnan(loss_train.item())): #TODO remove, for debugging
                print("Failed sim ", sim_index, "Batch ", n)
                exit()
            if (not eval):
                if OPTIMIZE_PHYSICS and random.random() < PROB_PHYSICS:
                    #Physics optimizacion
                    assert(loss_function_aux != None)
                    loss_train_aux = loss_function_aux(output,output,std)
                    optimizer.zero_grad()
                    loss_train_aux.backward() 
                    optimizer.step()
                else:
                    #Error optimization
                    optimizer.zero_grad() # set gradients to zero
                    loss_train.backward() # backwards pass
                    optimizer.step() # update model parameters
            avg_loss += loss_train.item()
            n += 1
            del x, y, output
    if n != 0: avg_loss /= n
    return avg_loss




#Train loop
modelName = model.__class__.__name__
if PREDICT_RESIDUAL: modelName += '_Res'
if OPTIMIZE_PHYSICS: modelName += '_Physics'
modelPath = "./savedModels/"+modelName+"/"
train_losses = []
eval_losses = []
test_losses = []
try:
    os.mkdir(modelPath)
except: pass
for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training...", ascii=False, ncols=50):
    #Training batch loop
    thisEpochTrainLoss = 0
    counterBatchTrain = 0
    bestEvalLoss = float('inf')
    #shuffle simulation order on each epoch? TODO

    train_losses.append(epoch_trainer(data_train_y,error_loss,loss_function_aux=physics_loss,eval=False))
    eval_losses.append(epoch_trainer(data_eval_y,error_loss,eval=True))
    test_losses.append(epoch_trainer(data_test_y,error_loss,eval=True))

    print(f' Epoch {epoch}, TrainL: {train_losses[-1]:.2e}, EvalL: {eval_losses[-1]:.2e}, TestL: {test_losses[-1]:.2e}')

    # Plotting the losses
    if (epoch >= 10 and epoch % 10 == 0):
        if True: 
            train_phys_loss = epoch_trainer(data_train_y,physics_loss,eval=True)
            eval_phys_loss = epoch_trainer(data_eval_y,physics_loss,eval=True)
            test_phys_loss = epoch_trainer(data_test_y,physics_loss,eval=True)
            print(f' TrainPhysicsL: {train_phys_loss:.2e}, EvalPhysicsPL: {eval_phys_loss:.2e}, TestPhysicsPL: {test_phys_loss:.2e}')

        fig, ax = plt.subplots(1,1,sharex=True,squeeze=True)
        ax.plot(train_losses[0:], label='Training MSE')
        ax.plot(eval_losses[0:], label='Evaluation MSE')
        ax.plot(test_losses[0:], label='Test MSE')
        ax.axhline(y=errorLoss_bicubic,color='black',label='Bicubic interpolation MSE')
        ax.set_title(label=modelName+" losses - LR: "+str(learning_rate))
        ax.legend()
        plt.savefig(modelPath+"losses_"+modelName+".png")
        plt.close()
    #Save best eval model after third epoch
    if epoch > 3 and eval_losses[-1] < bestEvalLoss:
        bestEvalLoss = eval_losses[-1]
        torch.save(model, modelPath+modelName+".pth")

#Save losses to files for future analisis and comparison between models
np.save(modelPath+"train_loss",np.array(train_losses))
np.save(modelPath+"eval_loss",np.array(eval_losses))
np.save(modelPath+"test_loss",np.array(test_losses))


