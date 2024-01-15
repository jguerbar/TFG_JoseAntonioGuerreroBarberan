#Exteral libraries imports
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
#Custom imports
from model import *
from losses import *
from enum import Enum
def epoch_trainer(model,data,loss_function,loss_function_aux=None,upscale_factor=4,
                    predict_res=True,batch_size=16):
    #Creation of instances of upsampler-downsampler
    bicubic = torch.nn.Upsample(scale_factor=upscale_factor, mode='bicubic')
    downsampler = torch.nn.AvgPool2d(upscale_factor)
    #Variable to store the average loss of all the minibatches for this epoch
    avg_loss = 0
    n = 0
    #Set the model for training
    model.train()
    optimize_aux_loss = loss_function_aux != None
    #Loop through data and train
    for sim_index in range(data.shape[0]):
        simulationData = data[sim_index]
        #Normalize
        std = np.std(data[sim_index],axis=(0,2,3),keepdims=True)
        simulationData = simulationData/std
        std = torch.tensor(std).float().to(device)

        for batch_index in range(0,data.shape[1],batch_size):
            y = torch.tensor(simulationData[batch_index:batch_index+batch_size,:,:,:]).float()
            x = downsampler(y)
            #Move to device, for efficient use of VRAM           
            x = x.to(device)
            y = y.to(device)

            #Randomly rotate matrix on spatial dimentions
            k = np.random.randint(low=0,high=4)
            x = torch.rot90(x,k,dims=(-1,-2))
            y = torch.rot90(y,k,dims=(-1,-2))

            output = model(x) # forwards pass
            if predict_res: #Reconstruct when modelling for PREDICT_RESIDUAL
                output = bicubic(x)+output
            loss_error = loss_function(output,y,std)
            optimizer.zero_grad() # set gradients to zero
            loss_error.backward(retain_graph=optimize_aux_loss) # backwards pass
            if optimize_aux_loss:
                #Extra loss optimization optimizacion
                loss_aux = loss_function_aux(output,y,std)
                loss_aux.backward() # backwards pass
            optimizer.step() # update model parameters
            avg_loss += loss_error.item()
            n += 1
            del x, y, output
    if n != 0: avg_loss /= n
    return avg_loss
def epoch_tester(model,data,loss_function,predict_res=True,upscale_factor=4,batch_size=16):
    #Creation of instances of upsampler-downsampler
    bicubic = torch.nn.Upsample(scale_factor=upscale_factor, mode='bicubic')
    downsampler = torch.nn.AvgPool2d(upscale_factor)

    #Variable to store the average loss of all the minibatches for this epoch
    avg_loss = 0
    n = 0

    #Set the model for evaluation
    model.eval()

    #Loop through data and train
    for sim_index in range(data.shape[0]):
        simulationData = data[sim_index]
        #Normalize
        std = np.std(data[sim_index],axis=(0,2,3),keepdims=True)
        simulationData = simulationData/std
        std = torch.tensor(std).float().to(device)
        for batch_index in range(0,data.shape[1],batch_size):
            y = torch.tensor(simulationData[batch_index:batch_index+batch_size,:,:,:]).float()
            x = downsampler(y)
            #Move to device, for efficient use of VRAM           
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                output = model(x) # forwards pass
                if predict_res: #Reconstruct when modelling for PREDICT_RESIDUAL
                    output = bicubic(x)+output
                loss_mse = loss_function(output,y,std)
            avg_loss += loss_mse.item()
            n += 1
            del x, y, output
    if n != 0: avg_loss /= n
    return avg_loss




if __name__=='__main__':
    Training_Modes = Enum('Mode', ['DIRECT_MSE', 'RESIDUAL_MSE', 'RESIDUAL_MSE_PHYSICS'])
    #Training parameters
    N_EPOCHS = 70              #Number of training epochs
    UPSAMPLE_FACTOR = 4         #Upsample factor
    BATCH_SIZE = 16             #Training Mini-Batch seize
    INPUT_CHANNELS = 3          #Number of channels, in this case representing physical variables
    MODE = Training_Modes.RESIDUAL_MSE

    """
    Load data with shape (N,T,C,W,H) where:
    N is the id of the simulation
    T is the timestep
    C is the channels u,v and p
    W is the grid width
    H is the grid height
    """
    data_train_y = np.load('.\\data\\trainData.npy')
    data_eval_y = np.load('.\\data\\evalData.npy')
    #data_test_y = np.load('.\\data\\testData.npy')

    print(data_train_y.shape)
    print(data_eval_y.shape)
    #print(data_test_y.shape)

    #Create model instance
    #model = NaiveTransConv(INPUT_CHANNELS, 512).to(device)
    #model = ED_TransConv(INPUT_CHANNELS).to(device)
    #model = ED_PixelShuffle(INPUT_CHANNELS,upscale_factor=4).to(device)
    #model = FSRCNN(INPUT_CHANNELS,d=56,s=12,m=10,scale_factor=UPSAMPLE_FACTOR).to(device)
    #model = SRCNN(INPUT_CHANNELS,64,32).to(device) 
    #model = VDSR(numInputChannels=INPUT_CHANNELS,deep_channels=64,num_layers=10,scale_factor=UPSAMPLE_FACTOR).to(device) 
    #model = ESPCN(INPUT_CHANNELS,INPUT_CHANNELS,64,UPSAMPLE_FACTOR).to(device)

    #Load model if needed
    
    model = torch.load('./savedModels/ED_TransConv_Res/ED_TransConv_Res.pth')
    train_losses = list(np.load('./savedModels/ED_TransConv_Res/train_loss.npy'))
    eval_losses = list(np.load('./savedModels/ED_TransConv_Res/eval_loss.npy'))
    
    #Model instance declaration
    #Optimizer parameters
    weight_decay = 1e-7
    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #Loss functions
    loss_func_error = MSE_loss()
    loss_func_physics = Navier_Stokes_informed_loss(rho=1, mu=0.00001, dt=1/32,dx=1/255,dy=1/255)

    bicubic = nn.Upsample(scale_factor=UPSAMPLE_FACTOR, mode='bicubic')
    downsampler = nn.AvgPool2d(UPSAMPLE_FACTOR)

    #Interpolation comparison losses
    for data_y in [data_train_y,data_eval_y]:
        errorLoss_bicubic = 0
        lphysicsInterp = 0
        lphysicsY = 0
        for sim_index in range(data_y.shape[0]):
            #Normalize input
            simulationData = data_y[sim_index]
            #Normalize
            std = np.std(data_y[sim_index],axis=(0,2,3),keepdims=True)
            simulationData = simulationData/std
            std = torch.tensor(std).float().to(device)
            y = torch.tensor(simulationData[:,:,:]).float()
            x = downsampler(y)
            y = y.to(device)
            x = x.to(device)
            bic_interpolation = bicubic(x)
            errorLoss_bicubic += loss_func_error(bic_interpolation,y,std).item()
            lphysicsY += loss_func_physics(y,y,std).item()
            lphysicsInterp += loss_func_physics(bic_interpolation, y, std).item()
        errorLoss_bicubic /= data_y.shape[0]
        lphysicsInterp /= data_y.shape[0]
        lphysicsY /= data_y.shape[0]
        print(f'BicubicMSE: {errorLoss_bicubic:.2e}, PhysicsLossInterp: {lphysicsInterp:.2e}')


    #Train loop
    modelName = model.__class__.__name__
    main_loss = loss_func_error
    aux_loss = None
    residual_pred = False
    if MODE == Training_Modes.RESIDUAL_MSE:
        modelName += '_Res'
        main_loss = loss_func_error
        aux_loss = None
        residual_pred = True
    if MODE == Training_Modes.RESIDUAL_MSE_PHYSICS:
        modelName += '_Res_Physics'
        main_loss = loss_func_error
        aux_loss = loss_func_physics
        residual_pred = True
    print(modelName)
    modelPath = "./savedModels/"+modelName+"/"
    #train_losses = []
    #eval_losses = []

    try:
        os.mkdir(modelPath)
    except: pass
    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training...", ascii=False, ncols=50):
        #Training batch loop
        bestEvalLoss = float('inf')
        train_losses.append(epoch_trainer(model,data_train_y,main_loss,loss_function_aux=aux_loss,
                                        upscale_factor=UPSAMPLE_FACTOR,predict_res=residual_pred,
                                        batch_size=BATCH_SIZE))
        eval_losses.append(epoch_tester(model,data_eval_y,main_loss,predict_res=residual_pred,
                                        upscale_factor=UPSAMPLE_FACTOR,batch_size=BATCH_SIZE))

        print(f' Epoch {epoch}, TrainL: {train_losses[-1]:.2e}, EvalL: {eval_losses[-1]:.2e}')

        # Plotting the losses
        if (epoch >= 10 and epoch % 10 == 0):
            train_phys_loss = epoch_tester(model,data_train_y,loss_func_physics,predict_res=MODE != Training_Modes.DIRECT_MSE,
                                            upscale_factor=UPSAMPLE_FACTOR,batch_size=BATCH_SIZE)
            eval_phys_loss = epoch_tester(model,data_eval_y,loss_func_physics,predict_res=MODE != Training_Modes.DIRECT_MSE,
                                            upscale_factor=UPSAMPLE_FACTOR,batch_size=BATCH_SIZE)
            print(f' TrainPhysicsL: {train_phys_loss:.2e}, EvalPhysicsPL: {eval_phys_loss:.2e}')

            fig, ax = plt.subplots(1,1,sharex=True,squeeze=True)
            ax.plot(train_losses, label='Training MSE')
            ax.plot(eval_losses, label='Evaluation MSE')
            ax.axhline(y=errorLoss_bicubic,color='black',label='Bicubic interpolation MSE')
            ax.set_title(label=modelName+" losses - LR: "+str(learning_rate))
            ax.legend()
            plt.savefig(modelPath+"losses_"+modelName+".png")
            plt.close()
            np.save(modelPath+"train_loss",np.array(train_losses))
            np.save(modelPath+"eval_loss",np.array(eval_losses))

        #Save best eval model after third epoch
        if epoch > 3 and eval_losses[-1] < bestEvalLoss:
            bestEvalLoss = eval_losses[-1]
            torch.save(model, modelPath+modelName+".pth")





