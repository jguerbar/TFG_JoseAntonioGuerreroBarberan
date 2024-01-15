import torch
from thop import profile
import os
from train import epoch_tester
from losses import MSE_loss
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

device = torch.device('cuda')
data_test_y = np.load('.\\data\\testData.npy')
print(data_test_y.shape)

root = ".\\savedModels"
savedModels = [ (item, os.path.join(root, item)) for item in os.listdir(root) if not os.path.isfile(os.path.join(root, item)) ]
#savedModels = [a for a in savedModels if a[0].find('Res') != -1 and a[0].find('Physics') == -1]

df = pd.DataFrame(columns=['ModelName','Params(K)','FLOPs(G)','Loss'])

for m in range(len(savedModels)):
    savedModel = savedModels[m]
    modelName = savedModel[0]
    path = savedModel[1]
    modelPath = path+"\\"+modelName+".pth"
    isRes = modelName.find('Res') != -1

    model = torch.load(modelPath).to(device)
    loss_error = epoch_tester(model,data_test_y,MSE_loss(),isRes,4,16)

    dsize = (2, 3, 255, 255)
    inputs = torch.randn(dsize).to(device)
    total_ops, total_params = profile(model, (inputs,), verbose=False)

    df.loc[len(df)] = modelName, total_params / (1000), 2*total_ops / (1000 ** 3), loss_error
df.to_csv('./complexity_table.csv')

fig, ax = plt.subplots(1,2,sharey=False,squeeze=True,figsize=(14,7),)
ax[0].scatter(df['Params(K)'],df['Loss'])
ax[1].scatter(df['FLOPs(G)'],df['Loss'])
for i in df.index:
    ax[1].annotate(df.loc[i]['ModelName'], (df.loc[i]['FLOPs(G)'], df.loc[i]['Loss']))
    ax[0].annotate(df.loc[i]['ModelName'], (df.loc[i]['Params(K)'], df.loc[i]['Loss']))
ax[0].set_ylabel('MSE')
ax[0].set_xlabel('Params(K)')
ax[1].set_ylabel('MSE')
ax[1].set_xlabel('FLOPs(G)')

plt.savefig('./complexity_figures.png')
plt.show()
print(df)

